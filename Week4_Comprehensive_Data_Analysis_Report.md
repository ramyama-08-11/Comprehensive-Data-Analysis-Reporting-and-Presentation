---
title: "Comprehensive Data Analysis Reporting and Presentation"
subtitle: "Week 4 Capstone | R Data Preparation, Visualization, and Predictive Modeling"
date: "3 October 2026"
---

# Executive Summary

This report brings together three preceding R analyses as a portfolio of complementary case studies: Palmer Archipelago penguins (data cleaning and exploratory analysis), U.S. economic indicators (visual communication), and the Wisconsin Diagnostic Breast Cancer dataset (statistical inference and classification). The studies use different datasets and answer different questions; they are integrated here through their analytical workflow, not by combining their records.

The penguin workflow retained all 344 source records, treated 19 missing cells, flagged no pooled IQR outliers, and found a strong pooled association between flipper length and body mass ($r=0.871$). Gentoo penguins had the largest mean body mass (5,076 g) and flipper length (217.2 mm) in the sample. The economic charts described monthly U.S. unemployment, savings, and population trends from 1967 to 2015; these are exploratory patterns, not causal or statistically tested effects. The breast-cancer analysis used four mean nuclear measurements in a logistic regression model. On a stratified 115-case holdout, it achieved 93.9% accuracy and an ROC AUC of 0.994 at a 0.50 threshold; repeated training-set cross-validation estimated mean AUC of 0.983.

Together, the projects demonstrate an end-to-end analytical story: validate the source, make data-quality decisions visible, choose visuals that fit the question, and evaluate a predictive model on data not used to fit it. Important limitations remain: the penguin summaries are pooled and exploratory; the economics report does not quantify uncertainty or establish causal relationships; and the breast-cancer benchmark is small and historic and is not suitable for clinical use. Recommendations focus on sensitivity analyses, clearer uncertainty communication, external validation, and audience-appropriate reporting.

# 1. Introduction and Scope

The Week 4 objective is to communicate prior work coherently to both technical and non-technical readers. The provided Week 1–3 materials address three distinct application domains:

| Case study | Data and purpose | Main analytical contribution |
|:--|:--|:--|
| Week 1 | 344 Palmer Archipelago penguin observations | Cleaning, missing-data handling, outlier screening, encoding, descriptive statistics, and exploratory plots |
| Week 2 | Monthly U.S. economic indicators, 1967–2015 | Time-series, relationship, distribution, and decade-comparison visualizations |
| Week 3 | 569 Wisconsin Diagnostic Breast Cancer observations | Prespecified hypothesis test and logistic-regression classification with resampling and holdout evaluation |

Because the data, outcomes, and units differ, a pooled analysis would be invalid. The synthesis instead compares the analytical decisions and lessons across the three cases. Findings are reported as descriptive unless a formal inferential test or predictive validation was actually performed. No new statistical tests or claims of causal impact are inferred from the supplied visual summaries.

# 2. Methodology

## 2.1 Common reporting approach

For each case, the report identifies the data source and question, summarizes the implemented R workflow, presents the original reported results and visual evidence, and distinguishes observations from limitations. Code excerpts are adapted from the supplied scripts to show the key analytical steps; the original runnable scripts and machine-readable results remain in their Week 1–3 project folders.

## 2.2 Case-study methods

**Penguins.** The base-R workflow imported the 344-by-8 CSV, inspected structure and summaries, counted missing values, screened observed morphology measurements using Tukey’s 1.5-IQR fences, and generated cleaned and scaled outputs. Missing numeric measurements were filled with field medians and marked with companion missingness flags; missing sex was represented as `Unknown`. Original measurement units were retained alongside min–max-scaled copies. Correlations used pairwise-complete observed values.

```r
for (field in numeric_traits) {
  missing_flag <- is.na(penguins_clean[[field]])
  median_value <- median(penguins_clean[[field]], na.rm = TRUE)
  penguins_clean[[paste0(field, "_was_missing")]] <- missing_flag
  penguins_clean[[field]][missing_flag] <- median_value
}
penguins_clean$sex[is.na(penguins_clean$sex)] <- "Unknown"
```

This is a transparent exploratory imputation strategy, not a substitute for modeling missingness or propagating imputation uncertainty.

**U.S. economic indicators.** The Week 2 script used `ggplot2::economics`, with `dplyr` and `lubridate` to derive year/month and rescale unemployment and population. It generated a dual-axis trend chart, a scatter plot of median unemployment duration against personal savings, a histogram, average unemployment by decade, and a population time series. A units audit of the supplied script identified a scaling/label mismatch: the source describes `unemploy` and `pop` in thousands, while the script divides them by 1,000 and 1,000,000, respectively, but labels the results as thousands and millions. The included original figures are retained as evidence of the Week 2 work, but their affected vertical-axis magnitudes should be corrected and the plots regenerated before quantitative use.

```r
econ <- ggplot2::economics %>%
  mutate(
    year = as.integer(format(date, "%Y")),
    month = month(date, label = TRUE),
    unemploy_millions = unemploy / 1000,
    population_millions = pop / 1000
  )

ggplot(econ, aes(x = date, y = unemploy_millions)) +
  geom_line(color = "#1f77b4", linewidth = 1.1) +
  labs(x = "Year", y = "Unemployed persons (millions)") +
  theme_minimal()
```

**Breast-cancer classification.** The Week 3 base-R analysis used the public UCI data, checked the expected 569-by-32 input and absence of missing values, and encoded benign as the reference class and malignant as the positive class. A reproducible, class-stratified 80/20 split created 454 training and 115 holdout observations. Repeated stratified five-fold cross-validation (five repeats) used training observations only; the final model was refitted on all training data and evaluated once on the holdout.

```r
predictors <- c(
  "radius_mean", "texture_mean", "smoothness_mean",
  "concave_points_mean"
)
model_formula <- reformulate(predictors, response = "diagnosis")
fit_model <- function(training_data) {
  glm(model_formula, data = training_data, family = binomial())
}

final_model <- fit_model(train)
test_probability <- predict(final_model, newdata = test, type = "response")
```

The probability cutoff was 0.50. Results include threshold-based metrics and threshold-independent ROC AUC, with Brier score for probability error. The cutoff was not optimized for a clinical or economic cost function.

# 3. Results

## 3.1 Week 1 — Data cleaning and penguin morphology

The source contained 344 records and eight fields. There were 19 missing cells: two in each of the four morphology variables and 11 in sex. Complete-case analysis would retain 333 records; median imputation and an explicit `Unknown` sex category retained all 344. No morphology measurement was flagged by the pooled IQR screen, so no values were removed. The cleaned table had no missing cells. Two records were missing all four measured traits, which makes their imputed morphology particularly uncertain.

Species composition was Adelie 152, Chinstrap 68, and Gentoo 124. The reported species means show marked differences: Gentoo had mean flipper length 217.2 mm and body mass 5,076 g; Adelie averaged 190.0 mm and 3,701 g; Chinstrap averaged 195.8 mm and 3,733 g. These are sample descriptions, not tested population differences.

![Measured penguin traits by species.](assets/yuva1/figure_1_distributions.png)

*Figure 1. Morphology distributions by species. Between-species variation is especially visible for flipper length and body mass.*

Flipper length and body mass had the strongest reported pooled pairwise correlation ($r=0.871$); bill length and flipper length were also positively associated ($r=0.656$). These pooled relationships may partly reflect differences among species. A follow-up should examine within-species patterns before interpreting the correlation biologically.

![Bill length and flipper length by species.](assets/yuva1/figure_2_bill_flipper.png)

*Figure 2. Bill length versus flipper length. The pooled trend is descriptive and can mask species-specific structure.*

![Penguin missingness counts.](assets/yuva1/figure_3_missingness.png)

*Figure 3. Missingness was limited but concentrated in the five fields shown; missing sex accounted for 11 of 19 cells.*

**Interpretation and implications.** Keeping an audit flag beside each imputed value preserves traceability. However, using global medians can compress variation and alter relationships, while missingness in all four measurements for two records cannot be informed by the other traits on those records. Before biological inference or prediction, compare complete-case and imputed estimates, investigate missingness by species/island/year, and consider species-stratified imputation or an appropriate multiple-imputation method.

## 3.2 Week 2 — U.S. economic indicators and visual storytelling

The supplied report describes monthly indicators over 1967–2015 and uses different chart types for different communication goals. The time-series chart compares unemployment volume with the personal savings rate; the scatter plot relates median unemployment duration to savings; the histogram summarizes the distribution of unemployment observations; the decade bars simplify period comparisons; and the population line provides long-run demographic context.

![Unemployment and savings over time.](assets/yuva2/plot1_economic_trends.png)

*Figure 4. A dual-axis view can orient readers to co-movement over time. Because the savings series is multiplied by ten to share the plotted scale, visual distance and apparent alignment depend on that arbitrary scaling.*

![Unemployment duration versus personal savings rate.](assets/yuva2/plot2_scatter_unemployment_vs_savings.png)

*Figure 5. The scatter plot is exploratory. Year is encoded by color, so long-run time trends may contribute to an apparent association; it does not by itself establish that longer unemployment causes lower savings.*

![Distribution of unemployment observations.](assets/yuva2/plot3_histogram_unemployment.png)

*Figure 6. The histogram summarizes the distribution of the monthly unemployment measure; the source chart’s “annual” subtitle is imprecise because the observations are monthly. Its unemployment axis label also needs unit correction.*

![Average unemployment by decade.](assets/yuva2/plot4_bar_unemployment_by_decade.png)

*Figure 7. Decade aggregation provides an accessible comparison but suppresses within-decade variation and recession timing.*

![Population growth time series.](assets/yuva2/plot5_population_growth.png)

*Figure 8. The population series supplies long-run context; population growth alone does not explain changes in unemployment.*

**Interpretation and implications.** The visualization set demonstrates how line charts, scatter plots, histograms, and aggregated bars answer complementary descriptive questions. The source-native unit mismatch means the displayed unemployment and population axis magnitudes are off by a factor of 1,000 relative to their labels; correct the transformations and regenerate all affected figures before quoting chart values. The plotted time-series shapes remain useful for orientation, but not their current labeled magnitudes. The supplied report interprets unemployment and savings as related dimensions of economic conditions, but no formal significance test, uncertainty interval, or causal design is provided. Accordingly, the relationship should be presented as a hypothesis for further analysis rather than a quantified effect. Dual-axis charts should be paired with separate panels or standardized series to reduce scaling ambiguity; the histogram label should explicitly say monthly observations.

## 3.3 Week 3 — Statistical inference and predictive modeling

The UCI dataset has 569 observations, 30 real-valued predictors, and 357 benign and 212 malignant diagnoses. The logistic regression deliberately used four prespecified mean measurements—radius, texture, smoothness, and concave points—rather than all 30 variables. This supports a compact teaching model but does not establish that these are the uniquely best predictors.

In the training set, mean concave points were 0.0261 for benign cases and 0.0884 for malignant cases. The estimated malignant-minus-benign difference was 0.0623 (95% CI 0.0566–0.0680); Welch’s two-sample test gave $p\approx1.06\times10^{-55}$. This is strong evidence of a difference in this benchmark sample, not evidence of causation or clinical usefulness. Normality tests indicated skew in both groups, so a rank-based or bootstrap sensitivity analysis would strengthen the inference.

![Distributions of the four model predictors by diagnosis.](assets/yuva3/feature_distributions.png)

*Figure 9. Radius and concave-points distributions show greater visual separation by diagnosis than texture and smoothness.*

Radius and concave points were strongly correlated (Pearson $r=0.825$); their VIFs were 4.77 and 6.82, respectively. This collinearity can make individual coefficient estimates less stable, even if predictive ranking remains strong. The coefficients should therefore not be read as causal effects.

| Metric | Repeated CV mean (SD) | Holdout |
|:--|--:|--:|
| Accuracy | 0.935 (0.022) | 0.939 |
| Sensitivity (malignant recall) | 0.897 (0.060) | 0.930 |
| Specificity | 0.957 (0.031) | 0.944 |
| Precision | 0.929 (0.047) | 0.909 |
| F1 | 0.911 (0.032) | 0.920 |
| Balanced accuracy | 0.927 (0.027) | 0.937 |
| ROC AUC | 0.983 (0.010) | 0.994 |
| Brier score (lower is better) | 0.050 (0.014) | 0.032 |

At the fixed 0.50 threshold, the holdout confusion matrix contained 68 true negatives, four false positives, three false negatives, and 40 true positives. The model therefore missed three of 43 malignant holdout cases. Sensitivity is important to report because an apparently high accuracy alone can obscure errors in the positive class. Repeated-fold standard deviations are descriptive variation across overlapping resamples, not formal confidence intervals.

![Holdout ROC curve.](assets/yuva3/holdout_roc_curve.png)

*Figure 10. The holdout AUC of 0.994 indicates strong ranking discrimination in this split; it does not imply clinical readiness.*

![Holdout confusion matrix at a 0.50 threshold.](assets/yuva3/holdout_confusion_matrix.png)

*Figure 11. Holdout classifications at the reported, unoptimized threshold.*

![Model diagnostics.](assets/yuva3/model_diagnostics.png)

*Figure 12. Residual, influence, and coarse calibration diagnostics help identify questions for further validation; they do not replace an external test cohort.*

The report includes an output snapshot for traceability:

![R console output snapshot from the model analysis.](assets/yuva3/r_console_results.png)

*Figure 13. Selected console evidence from the executed Week 3 R workflow.*

**Interpretation and implications.** The benchmark results support logistic regression as a strong baseline for this dataset. They do not justify clinical deployment: the data are historic and relatively small, the holdout is drawn from the same source, calibration evidence is limited, and false negatives have potentially asymmetric consequences. Any real-world assessment would require external and contemporary data, clinical oversight, carefully chosen thresholds, uncertainty estimates, and regulatory and ethical review.

# 4. Cross-Case Discussion

Across the three projects, the strongest common lesson is that trustworthy conclusions depend on transparent choices at every stage. Week 1 shows that preprocessing decisions (imputation, encoding, scaling, outlier review) change the analytical dataset and must be documented. Week 2 shows that visual encoding and aggregation can clarify a story, but can also imply a relationship more strongly than the evidence warrants. Week 3 shows why a model must be evaluated on observations withheld from fitting and why performance should be described with metrics that reflect different error types.

The evidence supports different strengths of conclusion:

1. **Descriptive:** penguin species means and observed economic time-series patterns summarize the supplied data.
2. **Inferential:** the concave-points Welch test quantifies evidence of a group mean difference in the training sample.
3. **Predictive:** cross-validation and a holdout estimate classification performance for a defined dataset and split.

These forms of evidence are not interchangeable. Statistical significance does not establish useful prediction; a high AUC does not establish causality; and a visually compelling association is not a significance test. Clear labels, units, sample sizes, denominators, and caveats make results more useful to a broad audience.

## Limitations and challenges

- The penguin analysis uses simple global median imputation and pooled outlier fences and does not perform inferential testing.
- The economic visual report does not provide numerical estimates, confidence intervals, or tests for its described associations; its dual-axis scaling and annual histogram subtitle require careful interpretation.
- The breast-cancer model is based on four selected features and one random holdout from the same benchmark source; it has not been externally validated.
- The three reports were prepared independently and use distinct datasets, so this capstone cannot claim a single, unified business or scientific outcome.
- Figures and reported metrics are taken from the supplied outputs; the present synthesis does not rerun the source analyses.

# 5. Recommendations and Future Directions

1. **Strengthen the penguin analysis:** compare imputed and complete-case summaries; document whether missingness varies by species, island, and year; inspect extreme values within species; and estimate species-adjusted relationships rather than relying only on pooled correlations.
2. **Improve economic inference:** replace or complement dual axes with aligned small multiples; quantify trends with appropriate time-series methods; account for serial dependence; report effect estimates and uncertainty; and avoid causal language unless a defensible causal design is introduced.
3. **Extend model validation:** obtain an independent contemporary dataset; report bootstrap uncertainty for performance; assess calibration and logit linearity; examine the seven holdout errors; evaluate decision thresholds using training-only predictions and an explicit cost/sensitivity goal; and compare regularized models under nested resampling.
4. **Improve reproducibility:** retain the raw source, scripts, session information, exported tables, and figure-generation code with each report. Record package/R versions and random seeds, and rerun the full workflow from a clean project directory before submission.
5. **Communicate for the audience:** state the question before the chart, define technical terms, label units, show sample sizes and uncertainty, and separate observed findings from recommendations.

# 6. Conclusion

The three-week portfolio demonstrates a practical progression from data quality to visual exploration to statistical modeling. The penguin work shows how to preserve records while documenting missingness and transformations. The economics work illustrates the value of varied chart forms while emphasizing that visual association is not causal proof. The breast-cancer work demonstrates a reproducible classification baseline with stratified resampling and a held-out evaluation, alongside appropriate warnings about benchmark data and clinical use.

The principal impact is methodological: better analysis becomes more credible when the audience can see how data were prepared, what the plots and metrics mean, and where the evidence stops. The next step is not simply a more complex model; it is more robust sensitivity analysis, external evidence, transparent uncertainty, and communication tailored to the decisions the analysis is meant to inform.

## Suggested 32-hour capstone work plan

The assignment recommends approximately 30–35 hours. The following 32-hour schedule is a planning allocation, not a claim that these hours were already spent:

| Activity | Hours |
|:--|--:|
| Review source projects, data provenance, and questions | 3 |
| Reproduce and audit cleaning and transformations | 5 |
| Verify descriptive analysis and visualizations | 5 |
| Review inference, resampling, and model diagnostics | 6 |
| Integrate narratives, figures, code, and caveats | 7 |
| Proofread, verify outputs, and prepare final DOCX | 6 |
| **Total planned** | **32** |

# References and Source Materials

1. Horst, A. M., Hill, A. P., & Gorman, K. B. (2020). *palmerpenguins: Palmer Archipelago (Antarctica) penguin data*. https://doi.org/10.5281/zenodo.3960218. Original ecological study: Gorman, K. B., Williams, T. D., & Fraser, W. R. (2014). *PLOS ONE, 9*(3), e90081. https://doi.org/10.1371/journal.pone.0090081.
2. Wickham, H. et al. *ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics*. Economic indicators used here are distributed as `ggplot2::economics`; the Week 2 script describes monthly U.S. data from 1967–2015. Dataset documentation: https://ggplot2.tidyverse.org/reference/economics.html.
3. Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). *Breast Cancer Wisconsin (Diagnostic)* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.
4. UCI Machine Learning Repository. *Breast Cancer Wisconsin (Diagnostic): dataset and variable information*. https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic.
5. Week 1 source report, `Week1_Data_Cleaning_R_Analysis.docx`, supporting `analysis.R`, `report.md`, source CSV, and exported tables/figures.
6. Week 2 source report, `Data_Visualization_Report.docx`, supporting `visualize_economics.R` and exported figures.
7. Week 3 source report, `Week3_Statistical_Analysis_Report.docx` and `.md`, supporting `analyze_wdbc.R`, data, exported tables, and evidence figures.
