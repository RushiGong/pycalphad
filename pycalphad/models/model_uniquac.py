"""Universal Quasi-Chemical Model (UNIQUAC) for PyCalphad."""
import itertools
from collections import OrderedDict

from symengine import (
    Float,
    Piecewise,
    S,
    StrictGreaterThan,
    Symbol,
    log,
)
from tinydb import where

import pycalphad.variables as v
from pycalphad import Model
from pycalphad.core.constants import MIN_SITE_FRACTION
from pycalphad.core.errors import DofError
from pycalphad.core.utils import wrap_symbol, unpack_species
from pycalphad.io.tdb import get_supported_variables

class ModelUNIQUAC(Model):
    """
    This model implements the Universal Quasi-Chemical Model (UNIQUAC) model [1]_
    for calculating the Gibbs energy of a phase. The symbolic expressions for UNIQUAC 
    are built based on the CALPHAD convention expression from Li et al. [2]_.
    
    Reference:
    ----------
    [1] D. S. Abrams and J. M. Prausnitz, “Statistical thermodynamics of liquid mixtures: a new expression
        for the excess Gibbs energy of partly or completely miscible systems,” AIChE journal, vol. 21, no. 1,
        pp. 116–128, 1975.
    [2] J. Li, B. Sundman, J. G. Winkelman, A. I. Vakis, and F. Picchioni, “Implementation of the UNIQUAC
        model in the OpenCalphad software,” Fluid Phase Equilibria, vol. 507, p. 112 398, 2020.
    """
    
    contributions = [
        ("ref", "reference_energy"),
        ("idmix", "ideal_mixing_energy"),
        ('xsmix', 'excess_mixing_energy'),
    ]
    
    def __init__(self, dbe, comps, phase_name, parameters=None):
        self._dbe = dbe
        self._endmember_reference_model = None
        self.components = set()
        self.constituents = []
        self.phase_name = phase_name.upper()
        phase = dbe.phases[self.phase_name]
        self.site_ratios = list(phase.sublattices)
        active_species = unpack_species(dbe, comps)
        for idx, sublattice in enumerate(phase.constituents):
            subl_comps = set(sublattice).intersection(active_species)
            self.components |= subl_comps

        self.site_ratios = tuple(self.site_ratios)

        # Verify that this phase is still possible to build
        is_pure_VA = set()
        for sublattice in phase.constituents:
            sublattice_comps = set(sublattice).intersection(self.components)
            if len(sublattice_comps) == 0:
                # None of the components in a sublattice are active
                # We cannot build a model of this phase
                raise DofError(
                    '{0}: Sublattice {1} of {2} has no components in {3}' \
                    .format(self.phase_name, sublattice,
                            phase.constituents,
                            self.components))
            is_pure_VA.add(sum(set(map(lambda s : getattr(s, 'number_of_atoms'),sublattice_comps))))
            self.constituents.append(sublattice_comps)
        if sum(is_pure_VA) == 0:
            #The only possible component in a sublattice is vacancy
            #We cannot build a model of this phase
            raise DofError(
                '{0}: Sublattices of {1} contains only VA (VACUUM) constituents' \
                .format(self.phase_name, phase.constituents))
        self.components = sorted(self.components)
        desired_active_pure_elements = [list(x.constituents.keys()) for x in self.components]
        desired_active_pure_elements = [el.upper() for constituents in desired_active_pure_elements
                                        for el in constituents]
        self.pure_elements = sorted(set(desired_active_pure_elements))
        self.nonvacant_elements = [x for x in self.pure_elements if x != 'VA']

        # Convert string symbol names to Symbol objects
        # This makes xreplace work with the symbols dict
        symbols = {Symbol(s): val for s, val in dbe.symbols.items()}

        if parameters is not None:
            self._parameters_arg = parameters
            if isinstance(parameters, dict):
                symbols.update([(wrap_symbol(s), val) for s, val in parameters.items()])
            else:
                # Lists of symbols that should remain symbolic
                for s in parameters:
                    symbols.pop(wrap_symbol(s))
        else:
            self._parameters_arg = None

        self._symbols = {wrap_symbol(key): value for key, value in symbols.items()}

        self.models = OrderedDict()
        self.build_phase(dbe)

        for name, value in self.models.items():
            # XXX: xreplace hack because SymEngine seems to let Symbols slip in somehow
            self.models[name] = self.symbol_replace(value, symbols).xreplace(get_supported_variables())

        self.site_fractions = sorted([x for x in self.variables if isinstance(x, v.SiteFraction)], key=str)
        self.state_variables = sorted([x for x in self.variables if not isinstance(x, v.SiteFraction)], key=str)

    
    def reference_energy(self, dbe):
        """
        Returns the weighted average of the endmember energies in symbolic form.
        """
        pure_param_query = (
            (where('phase_name') == self.phase_name) & \
            (where('parameter_type') == "UQCG") & \
            (where('constituent_array').test(self._purity_test))
        )
        phase = dbe.phases[self.phase_name]
        param_search = dbe.search
        pure_energy_term = self.redlich_kister_sum(phase, param_search, pure_param_query)
        return pure_energy_term / self._site_ratio_normalization


    def ideal_mixing_energy(self, dbe):
        #pylint: disable=W0613
        """
        Returns the ideal mixing energy in symbolic form.
        """
        phase = dbe.phases[self.phase_name]
        site_ratios = self.site_ratios
        ideal_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            ratio = site_ratios[subl_index]
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                # We lose some precision here, but this makes the limit behave nicely
                # We're okay until fractions of about 1e-12 (platform-dependent)
                mixing_term = Piecewise(
                    (sitefrac*log(sitefrac), StrictGreaterThan(sitefrac, sitefrac_limit)),
                    (0, True),
                    )
                ideal_mixing_term += (mixing_term*ratio)
        ideal_mixing_term *= (v.R * v.T)
        return ideal_mixing_term / self._site_ratio_normalization
    
    def _rx_i(self, dbe, species: v.Species):
        """
        Volume parameter for species i times mole fraction of species i.
        rx_i = r_i * x_i
        """
        terms=S.Zero
        phase=dbe.phases[self.phase_name]
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        uqcr_param_query=(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "UQCR") & \
            (where("constituent_array").test(self._array_validity))
        )
        params = dbe._parameters.search(uqcr_param_query)
        for subl_index, sublattice in enumerate(phase.constituents):
            sitefrac = v.SiteFraction(phase.name, subl_index, species)
            for param in params:
                if param["constituent_array"][0][0] == species:
                    r_i=param["parameter"]
                    terms=Piecewise(
                        (sitefrac*r_i,StrictGreaterThan(sitefrac, sitefrac_limit)),
                        (0, True)
                    )
        return terms
    
    def _rx_sum(self, dbe):
        """
        Returns the sum of r_i*x_i for all components in the phase.
        """
        rx_sum=S.Zero
        phase=dbe.phases[self.phase_name]
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                rx_sum+=self._rx_i(dbe, comp)
        return rx_sum
    
    def _phi_i(self, dbe, species: v.Species):
        """
        Returns the volume fraction of species i.
        phi_i = r_i*x_i / sum(r_j*x_j)
        """
        return self._rx_i(dbe, species)/self._rx_sum(dbe)
    
    
    def q_i(self, dbe, species: v.Species):
        """
        Surface area parameter of species i.
        """
        uqcq_param_query=(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "UQCQ") & \
            (where("constituent_array").test(self._array_validity))
        )
        params = dbe._parameters.search(uqcq_param_query)
        for param in params:
            if param["constituent_array"][0][0] == species:
                q=param["parameter"]           
        return q
    
    
    def _qx_i(self, dbe, species: v.Species):
        """
        Surface area parameter times mole fraction of species i.
        qx_i = q_i * x_i
        """
        terms=S.Zero
        phase=dbe.phases[self.phase_name]
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            sitefrac = v.SiteFraction(phase.name, subl_index, species)
            terms=Piecewise(
                (sitefrac*self.q_i(dbe, species), StrictGreaterThan(sitefrac, sitefrac_limit)),
                (0, True),
            )
        return terms
    
    def _qx_sum(self, dbe):
        """
        Return the sum of q_i*x_i for all components in the phase.
        """
        qx_sum=S.Zero
        phase=dbe.phases[self.phase_name]
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                qx_sum+=self._qx_i(dbe, comp)
        return qx_sum
    
    def _theta_i(self, dbe, species: v.Species):
        """
        Return the surface fraction of species i.
        theta_i = q_i*x_i / sum(q_j*x_j)
        """
        return self._qx_i(dbe, species)/self._qx_sum(dbe)
    
    
    def cmb_p1(self, dbe):
        """
        Computes the first part of combinatorial contribution to the mixing energy.
        sum_i (x_i * ln(phi_i / x_i))
        """
        phase = dbe.phases[self.phase_name]
        cmb_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                sitefrac = v.SiteFraction(phase.name, subl_index, comp)
                mixing_term = Piecewise(
                    (sitefrac*log(self._phi_i(dbe,comp)/sitefrac),StrictGreaterThan(sitefrac, sitefrac_limit)), 
                    (0, True)
                    ) 
                cmb_mixing_term += (mixing_term)
        cmb_mixing_term *= (v.R * v.T)
        return cmb_mixing_term / self._site_ratio_normalization
    
    def Z(self, dbe, species: v.Species):
        """
        Coordination number of species i.
        """
        uqcz_param_query=(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "UQCZ") & \
            (where("constituent_array").test(self._array_validity))
        )
        params = dbe._parameters.search(uqcz_param_query)
        for param in params:
            if param["constituent_array"][0][0] == species:
                z=param["parameter"]
        return z

    def cmb_p2(self, dbe):
        """
        Computes the second part of combinatorial contribution to the mixing energy.
        Z/2 * sum_i (x_i * q_i * ln(theta_i / phi_i)) 
        """
        phase = dbe.phases[self.phase_name]
        cmb_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                mixing_term = Piecewise(
                    (self.Z(dbe,comp)/2*sitefrac*self.q_i(dbe,comp)*log(self._theta_i(dbe,comp)/self._phi_i(dbe,comp)), StrictGreaterThan(sitefrac, sitefrac_limit)),
                    (0, True),
                    ) 
                cmb_mixing_term += (mixing_term)
        cmb_mixing_term *= (v.R * v.T)
        return cmb_mixing_term / self._site_ratio_normalization
    
    def combinatorial_contribution_excess_mixing_energy(self, dbe):
        """
        Returns the combinatorial contribution to the excess mixing energy in symbolic form.
        """
        Gcmb=self.cmb_p1(dbe)+self.cmb_p2(dbe)
        return Gcmb
    
    def _tau_ji(self, dbe, i: v.Species, j: v.Species):
        """
        Returns the pairwise interaction parameter for species i and j.
        """
        uqct_param_query=(
            (where("phase_name") == self.phase_name) & \
            (where("parameter_type") == "UQCT") & \
            (where("constituent_array").test(self._array_validity))
        )
        params = dbe._parameters.search(uqct_param_query)
        tau_ji=S.Zero
        for param in params:
            if i in param["constituent_array"][0]:
                if j in param["constituent_array"][0]:
                    comps = param["constituent_array"][0]
                    exs = param["exponents"]
                    combination=sorted(zip(exs, comps))
                    for ii in combination:
                        if ii[1] == i:
                            if ii[0]==1:
                                tau_ji=param['parameter']           
        return tau_ji
        
    def _pair_ij(self, i: v.Species):
        """
        Returns a list of pairs (i, j) where i is the first element and j is the second element.
        """
        pairs = list(itertools.permutations(self.components, 2))
        i_pair = list()
        for ii in pairs:
            if ii[0] == i:
                i_pair.append(ii)
        return i_pair
                
    def _rho_i(self, dbe, i: v.Species):
        """
        Returns the sum of theta_j * tau_ji for all j in the phase.
        rho_i = sum_j (theta_j * tau_ji) + theta_i
        """
        terms = S.Zero
        for i_pair in self._pair_ij(i):
            ti = i_pair[0]
            tj = i_pair[1]
            tau_ji = self._tau_ji(dbe, ti, tj)
            terms += tau_ji * self._theta_i(dbe, tj)
        terms += self._theta_i(dbe, i)
        return terms
    
    def residual_contribution_excess_mixing_energy(self, dbe):
        """
        Returns the residual contribution to the excess mixing energy in symbolic form.
        """
        phase = dbe.phases[self.phase_name]
        res_mixing_term = S.Zero
        sitefrac_limit = Float(MIN_SITE_FRACTION/10.)
        for subl_index, sublattice in enumerate(phase.constituents):
            active_comps = set(sublattice).intersection(self.components)
            for comp in active_comps:
                sitefrac = \
                    v.SiteFraction(phase.name, subl_index, comp)
                mixing_term =  Piecewise(
                    (sitefrac*self.q_i(dbe,comp)*log(self._rho_i(dbe,comp)), StrictGreaterThan(sitefrac, sitefrac_limit)), 
                    (0, True),
                    )
                res_mixing_term += (mixing_term)
        res_mixing_term *= (v.R * v.T * (-1))
        return res_mixing_term / self._site_ratio_normalization
    
    def excess_mixing_energy(self, dbe):
        """
        Returns the excess mixing energy in symbolic form, which is the sum of the
        residual and combinatorial contributions.
        """
        return self.residual_contribution_excess_mixing_energy(dbe)+self.combinatorial_contribution_excess_mixing_energy(dbe)
        
    def build_phase(self, dbe):
        """
        Generate the symbolic form of all the contributions to this phase.

        Parameters
        ----------
        dbe : 'pycalphad.io.Database'
        """
        self.models.clear()
        for key, value in self.__class__.contributions:
            self.models[key] = S(getattr(self, value)(dbe))