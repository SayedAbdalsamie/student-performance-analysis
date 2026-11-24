"""Student performance: data cleaning, statistical analysis, results write-up.
Data: UCI Student Performance (Cortez & Silva, 2008), CC BY 4.0."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
log = []  # cleaning log

# ---------- 1. Load ----------
frames = []
for subj, f in [("Math", "data/raw/student-mat.csv"), ("Portuguese", "data/raw/student-por.csv")]:
    d = pd.read_csv(f, sep=";", quotechar='"', skipinitialspace=True)
    d["subject"] = subj
    frames.append(d)
    log.append(f"Loaded {subj}: {d.shape[0]} rows, {d.shape[1]-1} columns.")
df = pd.concat(frames, ignore_index=True)

# ---------- 2. Clean ----------
for c in df.select_dtypes("object").columns:
    df[c] = df[c].astype(str).str.strip().str.strip('"')
for c in ["G1", "G2", "G3"]:   # stored as quoted text in the raw files
    df[c] = pd.to_numeric(df[c], errors="coerce")
log.append("Grades G1, G2, G3 were stored as quoted text; converted to numbers.")
log.append(f"Missing values after conversion: {int(df.isna().sum().sum())}.")
dups = int(df.drop(columns="subject").duplicated().sum())
log.append(f"Fully duplicated rows across both subjects (excluding subject): {dups}. Kept: the same student appears in both subject files by design, and no duplicate exists within a file ({int(df.duplicated().sum())} exact duplicates).")
bin_cols = ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]
for c in bin_cols:
    df[c] = df[c].map({"yes": 1, "no": 0})
log.append("Yes/no columns encoded as 1/0: " + ", ".join(bin_cols) + ".")
# range checks
rng = {"age":(15,22),"Medu":(0,4),"Fedu":(0,4),"traveltime":(1,4),"studytime":(1,4),"failures":(0,4),
       "famrel":(1,5),"freetime":(1,5),"goout":(1,5),"Dalc":(1,5),"Walc":(1,5),"health":(1,5),"G1":(0,20),"G2":(0,20),"G3":(0,20)}
bad = {c:int(((df[c]<lo)|(df[c]>hi)).sum()) for c,(lo,hi) in rng.items()}
log.append("Out-of-range values per column: " + (", ".join(f"{k}={v}" for k,v in bad.items() if v) or "none") + ".")
# outliers
q1,q3 = df.absences.quantile([.25,.75]); iqr=q3-q1; hi=q3+1.5*iqr
n_out = int((df.absences>hi).sum())
log.append(f"Absences: {n_out} values above the IQR fence ({hi:.0f}); kept as real observations, analysed with rank-based methods and a log transform.")
zero = df[df.G3==0]
log.append(f"Final grade G3 = 0 for {len(zero)} students ({len(zero)/len(df):.1%}); they have G2 > 0 in {int((zero.G2>0).sum())} cases, i.e. likely dropped out of the final exam. Kept, and results are also shown without them.")
df["log_absences"] = np.log1p(df.absences)
df["pass"] = (df.G3 >= 10).astype(int)
df.to_csv("data/clean/student_clean.csv", index=False)
log.append(f"Clean dataset: {df.shape[0]} rows x {df.shape[1]} columns, saved to data/clean/student_clean.csv.")

R = {}  # results
def desc(s): return f"{s.mean():.2f} (SD {s.std():.2f})"

# ---------- 3. Descriptives ----------
R["n"] = len(df); R["pass_rate"] = df["pass"].mean()
bysub = df.groupby("subject").G3.agg(["count","mean","std","median"])
# figures
fig, ax = plt.subplots(1,2, figsize=(10,4))
sns.histplot(data=df, x="G3", hue="subject", bins=21, ax=ax[0]); ax[0].set_title("Distribution of final grade (G3)")
sns.boxplot(data=df, x="subject", y="G3", ax=ax[1]); ax[1].set_title("Final grade by subject")
plt.tight_layout(); plt.savefig("figures/fig1_grade_distribution.png", dpi=150); plt.close()

# ---------- 4. Tests ----------
tests = []
def two_group(var, label):
    a = df[df[var]==df[var].unique()[0]].G3; 
    vals = sorted(df[var].unique())
    g = [df[df[var]==v].G3 for v in vals]
    u,p = stats.mannwhitneyu(g[0], g[1], alternative="two-sided")
    t,pt = stats.ttest_ind(g[0], g[1], equal_var=False)
    d = (g[1].mean()-g[0].mean())/np.sqrt((g[0].var()+g[1].var())/2)
    tests.append((label, f"{vals[0]}: {desc(g[0])} vs {vals[1]}: {desc(g[1])}", f"Welch t={t:.2f}, Mann-Whitney p={p:.3g}", f"Cohen d={d:.2f}", p))
two_group("sex","Sex"); two_group("higher","Wants higher education (0/1)"); two_group("internet","Internet at home (0/1)")
two_group("schoolsup","Extra school support (0/1)"); two_group("address","Home address (R/U)"); two_group("subject","Subject")
# multi-group
for var,label in [("studytime","Weekly study time (1-4)"),("failures","Past class failures"),("Medu","Mother's education (0-4)"),("Walc","Weekend alcohol use (1-5)")]:
    groups=[x.G3.values for _,x in df.groupby(var) if len(x)>=5]
    h,p = stats.kruskal(*groups)
    eps = (h-len(groups)+1)/(len(df)-len(groups))
    tests.append((label, "; ".join(f"{k}: {x.G3.mean():.1f}" for k,x in df.groupby(var)), f"Kruskal-Wallis H={h:.1f}", f"epsilon2={eps:.3f}", p))
# chi-square pass vs
for var,label in [("higher","Wants higher education"),("schoolsup","Extra school support"),("romantic","In a relationship")]:
    ct = pd.crosstab(df[var], df["pass"]); chi,p,_,_ = stats.chi2_contingency(ct)
    v = np.sqrt(chi/ct.values.sum())
    tests.append((f"Pass (G3>=10) vs {label}", "pass rate " + ", ".join(f"{i}: {ct.loc[i,1]/ct.loc[i].sum():.0%}" for i in ct.index), f"chi2={chi:.1f}", f"Cramer V={v:.2f}", p))
# correlations
cors=[]
for c in ["G1","G2","studytime","failures","absences","Medu","Fedu","goout","Dalc","Walc","health","freetime","age","traveltime"]:
    r,p = stats.spearmanr(df[c], df.G3); cors.append((c,r,p))
cors = pd.DataFrame(cors, columns=["variable","spearman_rho","p"]).sort_values("spearman_rho", key=abs, ascending=False)
# Holm correction
allp = [t[4] for t in tests]+list(cors.p); order=np.argsort(allp); m=len(allp); adj=np.empty(m); run=0
for rank,i in enumerate(order):
    run=max(run,(m-rank)*allp[i]); adj[i]=min(1,run)
tests = [t+(adj[i],) for i,t in enumerate(tests)]
cors["p_holm"] = adj[len(tests):]
cors.to_csv("reports/spearman_correlations.csv", index=False)
pd.DataFrame(tests, columns=["comparison","group_summary","test","effect_size","p","p_holm"]).to_csv("reports/hypothesis_tests.csv", index=False)

# figures
fig, ax = plt.subplots(1,3, figsize=(14,4))
sns.boxplot(data=df, x="studytime", y="G3", ax=ax[0]); ax[0].set_title("G3 by weekly study time")
sns.boxplot(data=df, x="failures", y="G3", ax=ax[1]); ax[1].set_title("G3 by past failures")
sns.regplot(data=df, x="log_absences", y="G3", scatter_kws={"alpha":.25,"s":12}, ax=ax[2]); ax[2].set_title("G3 vs log(1+absences)")
plt.tight_layout(); plt.savefig("figures/fig2_key_relationships.png", dpi=150); plt.close()
num = ["G1","G2","G3","studytime","failures","log_absences","Medu","Fedu","goout","Walc","health","age"]
plt.figure(figsize=(8,6.5)); sns.heatmap(df[num].corr(method="spearman"), annot=True, fmt=".2f", cmap="coolwarm", center=0, annot_kws={"size":7})
plt.title("Spearman correlations"); plt.tight_layout(); plt.savefig("figures/fig3_correlation_heatmap.png", dpi=150); plt.close()

# ---------- 5. Regression (OLS via numpy, no G1/G2 so the model explains *background*, not prior grades) ----------
def ols(X, y, names):
    X = np.column_stack([np.ones(len(X)), X]); beta,*_ = np.linalg.lstsq(X,y,rcond=None)
    res = y-X@beta; n,k = X.shape; s2 = res@res/(n-k); cov = s2*np.linalg.inv(X.T@X)
    se = np.sqrt(np.diag(cov)); t = beta/se; p = 2*stats.t.sf(abs(t), n-k)
    ci = np.column_stack([beta-1.96*se, beta+1.96*se]); r2 = 1-res@res/((y-y.mean())@(y-y.mean()))
    return pd.DataFrame({"coef":beta,"se":se,"t":t,"p":p,"ci_low":ci[:,0],"ci_high":ci[:,1]}, index=["intercept"]+names), r2, 1-(1-r2)*(n-1)/(n-k)
d2 = df[df.G3>0]
feat = ["studytime","failures","log_absences","higher","Medu","Fedu","goout","Walc","internet","age"]
def build(d):
    X = d[feat].astype(float).copy(); X["male"]=(d.sex=="M").astype(float); X["portuguese"]=(d.subject=="Portuguese").astype(float)
    return X
for tag,d in [("all",df),("excl_zero",d2)]:
    X=build(d); tab,r2,ar2 = ols(X.values, d.G3.values.astype(float), list(X.columns))
    tab.round(4).to_csv(f"reports/regression_{tag}.csv"); R[f"r2_{tag}"]=(r2,ar2,len(d)); R[f"reg_{tag}"]=tab
# model incl. G2 for context
X = build(df); X["G2"]=df.G2; tab,r2,ar2 = ols(X.values, df.G3.values.astype(float), list(X.columns)); R["r2_g2"]=(r2,ar2); R["reg_g2"]=tab
# residual plot
X=build(df); tab_,_,_ = ols(X.values, df.G3.values.astype(float), list(X.columns))
Xm=np.column_stack([np.ones(len(X)),X.values]); fit=Xm@tab_.coef.values; resid=df.G3.values-fit
fig,ax=plt.subplots(1,2,figsize=(10,4)); ax[0].scatter(fit,resid,s=8,alpha=.4); ax[0].axhline(0,c="k"); ax[0].set_title("Residuals vs fitted"); stats.probplot(resid,plot=ax[1]); ax[1].set_title("Residual Q-Q plot")
plt.tight_layout(); plt.savefig("figures/fig4_regression_diagnostics.png", dpi=150); plt.close()

# ---------- 6. Write-up ----------
def pfmt(p): return "<0.001" if p<0.001 else f"{p:.3f}"
L=[]; A=L.append
A("# Results: What is associated with final grades in secondary school?\n")
A("*Dataset: UCI Student Performance (Cortez & Silva, 2008), CC BY 4.0. Full code: `analysis.py`. Every number below is produced by that script.*\n")
A("## 1. Data and cleaning\n"); [A(f"- {x}") for x in log]
A(f"\nThe analysis sample is {R['n']} student-subject records ({int((df.subject=='Math').sum())} Math, {int((df.subject=='Portuguese').sum())} Portuguese). Overall {R['pass_rate']:.1%} reached a pass grade (G3 >= 10 on the 0-20 scale).\n")
A("## 2. Descriptive results\n"); A("| Subject | n | Mean G3 | SD | Median |\n|---|---|---|---|---|")
for s,r in bysub.iterrows(): A(f"| {s} | {int(r['count'])} | {r['mean']:.2f} | {r['std']:.2f} | {r['median']:.0f} |")
A("\n![Grade distribution](../figures/fig1_grade_distribution.png)\n")
A("**Figure 1.** G3 is slightly left-skewed with a small spike at 0 (students who did not sit the final assessment).\n")
A("## 3. Group comparisons\n")
A("Binary predictors use Welch t-tests with Mann-Whitney U as a robustness check, multi-level predictors use Kruskal-Wallis, and pass/fail uses chi-square. P-values in the last column are Holm-adjusted across all tests in this report.\n")
A("| Comparison | Group summary (mean G3 or pass rate) | Test | Effect size | p | p (Holm) |\n|---|---|---|---|---|---|")
for t in tests: A(f"| {t[0]} | {t[1]} | {t[2]} | {t[3]} | {pfmt(t[4])} | {pfmt(t[5])} |")
A("\n![Key relationships](../figures/fig2_key_relationships.png)\n")
A("**Figure 2.** Final grade by study time, past failures and (log) absences.\n")
A("## 4. Correlations\n"); A("Spearman rank correlations with G3 (sorted by strength):\n"); A("| Variable | rho | p | p (Holm) |\n|---|---|---|---|")
for _,r in cors.iterrows(): A(f"| {r.variable} | {r.spearman_rho:.2f} | {pfmt(r.p)} | {pfmt(r.p_holm)} |")
A("\n![Correlation heatmap](../figures/fig3_correlation_heatmap.png)\n")
A("## 5. Regression\n")
A("Ordinary least squares on G3 using background and behaviour variables only (study time, failures, log absences, wants higher education, parents' education, going out, weekend alcohol, internet, age, sex, subject). Prior grades G1/G2 are left out on purpose so the model describes background factors instead of repeating earlier grades.\n")
for tag,title in [("all","All records"),("excl_zero","Excluding G3 = 0")]:
    r2,ar2,n=R[f"r2_{tag}"]; A(f"**{title}** (n={n}, R2={r2:.3f}, adjusted R2={ar2:.3f})\n"); A("| Term | Coef | 95% CI | p |\n|---|---|---|---|")
    for k,r in R[f"reg_{tag}"].iterrows(): A(f"| {k} | {r.coef:.3f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | {pfmt(r.p)} |")
    A("")
r2,ar2=R["r2_g2"]; A(f"For context, adding the mid-year grade G2 raises R2 to {r2:.3f} (adjusted {ar2:.3f}): prior performance is by far the strongest predictor.\n")
A("![Diagnostics](../figures/fig4_regression_diagnostics.png)\n")
A("**Figure 4.** Residuals are roughly symmetric, with a visible tail from the zero-grade students, which is why the model is repeated without them.\n")
# dynamic key findings
top = cors[~cors.variable.isin(["G1","G2"])].head(3)
A("## 6. Summary of findings\n")
A(f"- Prior grades dominate: Spearman rho with G3 is {cors.set_index('variable').loc['G2','spearman_rho']:.2f} for G2 and {cors.set_index('variable').loc['G1','spearman_rho']:.2f} for G1.")
A("- Among non-grade variables the strongest associations with G3 are " + ", ".join(f"{r.variable} (rho={r.spearman_rho:.2f})" for _,r in top.iterrows()) + ".")
sig=[t[0] for t in tests if t[5]<0.05]
A("- Comparisons that stay significant after Holm correction: " + ("; ".join(sig) if sig else "none") + ".")
A("\n## 7. Limitations\n- Observational data from two schools in Portugal (2005-2006); results show association, not cause.\n- Some students appear in both subject files, so records are not fully independent.\n- Self-reported variables (alcohol, study time) can be biased.\n- G3 = 0 mixes true zero performance with missing the final exam.\n- Grades are ordinal; rank-based tests were used for that reason, and OLS is shown for interpretability.")
open("reports/results_report.md","w").write("\n".join(L))
open("reports/cleaning_log.md","w").write("# Cleaning log\n\n"+"\n".join("- "+x for x in log)+"\n")
print("done", R["n"], R["pass_rate"]); print("\n".join(log))
