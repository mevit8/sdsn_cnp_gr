Climate Neutrality Pathways for Greece Scenario Selector Dashboard is based on <a href="https://www.unsdsn.org/our-work/sdsn-global-climate-hub">SDSN Global Climate Hub</a> Climate Neutrality Pathways for Greece: <a href="https://unsdsn.globalclimatehub.org/wp-content/uploads/2025/06/REPORT_GCHmodels_SDSNscenario_Greece__2.pdf">Integrated Assessment and Decision Support Tool Report</a>: June 2025

## The Simulation considers two scenarios: 

**Do-nothing scenario (business-as-usual - BAU)** which assumes that the current trends will continue applying until 2050; 

**NCNC (National Climate Neutrality Commitments) scenario** which assumes that the main sectoral climate-neutrality policies are jointly implemented. This includes practically the Greek National Energy and Climate Plan (NECP), the Greek Common Agricultural Policy (CAP), the shipping regulation targets as set by IMO and EU Emissions Trading System (ETS), and the River Basin Management Plans (RBMPs).

*Other policies that are not considered in this study concern mostly economic measures, which were not simulated because this example does not include economic models.
Ecosystems Economic Valuation & Beyond-GDP models, and General Computable Equilibrium (GCE) models are included in our future research plans.*

## Model Integration

**Conceptual flowchart showing how the different models are combined**

<p align="center">
  <img src="https://raw.githubusercontent.com/mevit8/sdsn_cnp_gr/main/content/modelling_flowchart.png" width="800">
</p>

List of models used and their role in our framework:

- FABLE Calculator (Mosnier et al., 2020) for the potential evolution of food and land-use systems; 

- The Low Emissions Analysis Platform (LEAP) (Heaps, 2022) for the simulation of the energy consumption and the associated GHG emissions of multiple pollutants; 

- MaritimeGCH for the simulation of the shipping sector’s climate-neutrality (Alamanos, Koundouri, et al., 2024)

- WaterRequirements accounting tool (Alamanos & Koundouri, 2024) for the estimation of the water requirements of the studied sectors

- LandReqCalcGCH model to estimate the land requirements for any potentially additional renewable energy production units,

- The BiofuelGCH Calculator to estimate the potential for biofuels' production using crop residues.

- All the models run under a common simulation period, 2020 to 2050, at an annual time-step. 

**References**
Koundouri P., Alamanos A., Arampatzidis I., Devves S., Dellis K., Deranian C., Nisiforou O. (2025). ClimateNeutrality Pathways for Greece: Integrated Assessment and Decision Support Tool. Report, UN SDSNGlobal Climate Hub. June 2025, Athens, Greece. DOI: 10.13140/RG.2.2.15107.62248

Alamanos, A. (2025). MaritimeGCH [Computer software]. https://github.com/Alamanos11/MaritimeGCH

Alamanos, A., & Koundouri, P. (2024). Estimating the water requirements per sector in Europe.
https://www.researchgate.net/profile/Angelos-
Alamanos/publication/386245556_Estimating_the_water_requirements_per_sector_in_Europe/links/6
749d5873d17281c7deacbd1/Estimating-the-water-requirements-per-sector-in-
Europe.pdf?__cf_chl_tk=fHpUHdaHJiBdO64MqGFr6VdICQGVzayXeMIMNj0xgg8-1738254361-1.0.1.1-
gml.ztb2RIAZtV8FEt_VNzJfQjcHuYhn1AAaEIVwPNk

Heaps, C.G., (2022). LEAP: The Low Emissions Analysis Platform (Version 2024.1.1.15) [Computer software]. Stockholm Environment Institute. https://leap.sei.org

Mosnier, A., Penescu, L., Pérez-Guzmán, K., Steinhauser, J., Thomson, M., Douzal, C., & Poncet, J. (2020). FABLE Calculator 2020 update. https://doi.org/10.22022/ESM/12-2020.16934