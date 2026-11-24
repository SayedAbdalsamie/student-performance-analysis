# Student Performance Analysis: Cleaning, Statistics and Report

End-to-end data analysis of secondary-school student grades: data cleaning, descriptive statistics, hypothesis tests with multiple-testing correction, regression, figures and a written report.

**Full report:** [`reports/Student_Performance_Analysis_Report.pdf`](reports/Student_Performance_Analysis_Report.pdf) (Markdown version: [`reports/Student_Performance_Analysis_Report.md`](reports/Student_Performance_Analysis_Report.md))

## Dataset and licence
UCI Machine Learning Repository, "Student Performance" dataset, by Paulo Cortez and Alice Silva (2008). It is released under the **Creative Commons Attribution 4.0 (CC BY 4.0)** licence. The raw files in `data/raw/` are unchanged copies; `data/clean/student_clean.csv` is my cleaned version. Please credit the original authors if you reuse the data.

> P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira (Eds.), Proceedings of 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), pp. 5-12, Porto, Portugal, April 2008.

## What the project does
1. **Cleaning:** converts text-stored grades, encodes yes/no fields, checks ranges and duplicates, documents outliers and the 53 zero-grade records (`reports/cleaning_log.md`).
2. **Tests:** Welch t-test with Mann-Whitney check, Kruskal-Wallis, chi-square, Spearman correlations, effect sizes, Holm correction.
3. **Regression:** OLS on background factors, with and without zero-grade students, plus a model with the mid-year grade for context. Diagnostics plotted.
4. **Report:** executive summary, interpretation, limitations and follow-up suggestions.

## Headline results
- Earlier grades explain most of the final grade (Spearman rho 0.95 with G2).
- Past failures are the strongest non-grade factor; wanting higher education shows a large gap (Cohen d = 0.89).
- Background variables alone explain about a quarter of the variance, so they suit screening, not individual prediction.

## Structure
```
analysis.py            cleaning, tests, regression, figures
build_report.py        builds the written report and PDF
data/raw, data/clean   original and cleaned data
figures/               4 figures
reports/               report (PDF/MD), test tables, regression tables
```

## Run
```
pip install pandas numpy scipy matplotlib seaborn
python analysis.py
python build_report.py   # needs pandoc and wkhtmltopdf for the PDF
```
Code: MIT licence. Data: CC BY 4.0 (see above).
