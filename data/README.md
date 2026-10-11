# MetroPT-3 Dataset Information

## Provenance and Citation
The dataset used in this project is the **MetroPT-3 dataset**, published on the UCI Machine Learning Repository and described in *Nature Scientific Data*:

- **Citation**: Veloso, B., Malheiro, B., Burguillo, J.C., Ribeiro, R.P., & Gama, J. (2022). *The MetroPT dataset for predictive maintenance*. Scientific Data, 9(1), 748. https://doi.org/10.1038/s41597-022-01877-3
- **UCI Repository Link**: [https://archive.ics.uci.edu/dataset/791/metropt+3+dataset](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).

## Hardware Context
- **Machine**: Knorr-Bremse VV120-T air compressor installed on Metro do Porto train unit PM179Tc.
- **Subsystem**: Air Production Unit (APU) supplying compressed air for braking, suspension, and door operations.
- **Timespan**: February 2020 through September 2020.
- **Sampling Frequency**: 1 Hz raw aggregated to 10-second intervals (~1.51 million rows).

## Sensor Description

### Analogue Sensors (Continuous)
1. **TP2** (bar): Pressure measurement at the compressor outlet.
2. **TP3** (bar): Pressure measurement at the pneumatic panel.
3. **H1** (bar): Pressure measurement after the desiccant air drying unit.
4. **DV_pressure** (bar): Differential pressure drop across the dryer purge valve.
5. **Reservoirs** (bar): Air pressure inside the main auxiliary reservoir.
6. **Oil_temperature** (°C): Temperature of lubricating oil inside the compressor crankcase.
7. **Motor_current** (A): Three-phase electrical current consumed by the compressor electric motor.

### Digital Signals (Binary / Discrete)
1. **COMP**: Compressor run command (1 = running, 0 = idle).
2. **DV_eletric**: Electrical command triggering the drain/purge valve.
3. **Towers**: Active drying tower selector (0 = Tower 1, 1 = Tower 2).
4. **MPG**: Motor-compressor pressure governor signal (cuts in at ~7.5 bar, cuts out at ~10 bar).
5. **LPS**: Low pressure switch alert (1 = low pressure condition).
6. **Pressure_switch**: Compressor cut-in / cut-out pressure switch signal.
7. **Oil_level**: Oil level float sensor (1 = normal level, 0 = low oil alert).
8. **Caudal_impulses**: Air flow pulse indicator measuring volumetric delivery.

## Integrity Rules
- The raw CSV in `data/raw/MetroPT3(AirCompressor).csv` is read-only and **must never be modified or overwritten**.
- All processed artifacts are generated reproducibly by code in `src/` and stored in `data/processed/`.
