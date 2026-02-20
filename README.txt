1. Dataset Overview
Title: Daily Population Flow Networks of Chinese Prefectural Cities from 2020 to 2025 (DPFNet-CPC) 
Temporal Coverage: 1 January 2020 to 31 December 2025 
Spatial Coverage: 366 prefecture-level administrative units in mainland China 
Unit: Absolute person-times (calibrated from relative LBS indices) 

2. Data Record Description
The dataset is organized by year and stored in OpenXML format (.xlsx). Each file represents a daily Origin-Destination (OD) matrix.
- File naming convention: DPFNet_CPC_YYYYMMDD.xlsx
- Lookup Table: Unit_Lookup.xlsx contains the mapping between Administrative Division (AD) codes, Pinyin, and Chinese city names.

3. Data Gaps and Missingness
Due to technical maintenance and unavailability of the raw source indices (Baidu Migration Platform), the following dates are missing from the dataset:
Year 2020 (145 days total missing)
 - May 7 to June 24: Data gap during the early post-lockdown recovery phase.
 - June 28 to September 21: Extended maintenance period of the raw data source.
Year 2021 (4 days total missing) 
- July 8, August 12, August 25, and November 26.
Year 2022 to 2025
- Zero missing days. The record is continuous for this period.

4. Methodological Summary
The DPFNet-CPC dataset was produced using the Multi-scale Anchor-Calibrated Mobility (MAC-Mobility) framework. This framework synchronizes high-resolution Location-Based Services (LBS) indices with macro-scale transport anchors from the Ministry of Transport of China using an adaptive CatBoost calibration model (R^2 = 0.939，MAPE=4.2%).

5. Usage Recommendations
- Handling Gaps: Users should be cautious when performing time-series forecasting or trend analysis during 2020. We recommend using seasonal decomposition or historical averages from 2021-2025 to account for these gaps.
- Spatial Alignment: All city-to-city flows are anchored to the 2022 administrative division standards. No significant administrative restructuring occurred at the prefecture level during the study period, ensuring longitudinal consistency.
- Identification: Please use AD Codes as the primary key for spatial merging to avoid phonetic translation ambiguity (e.g., distinguishing Suzhou in Anhui from Suzhou in Jiangsu).

6. Citation
If you use this dataset in your research, please cite the following Data Descriptor: