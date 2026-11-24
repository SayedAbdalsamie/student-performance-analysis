# Cleaning log

- Loaded Math: 395 rows, 33 columns.
- Loaded Portuguese: 649 rows, 33 columns.
- Grades G1, G2, G3 were stored as quoted text; converted to numbers.
- Missing values after conversion: 0.
- Fully duplicated rows across both subjects (excluding subject): 0. Kept: the same student appears in both subject files by design, and no duplicate exists within a file (0 exact duplicates).
- Yes/no columns encoded as 1/0: schoolsup, famsup, paid, activities, nursery, higher, internet, romantic.
- Out-of-range values per column: none.
- Absences: 54 values above the IQR fence (15); kept as real observations, analysed with rank-based methods and a log transform.
- Final grade G3 = 0 for 53 students (5.1%); they have G2 > 0 in 33 cases, i.e. likely dropped out of the final exam. Kept, and results are also shown without them.
- Clean dataset: 1044 rows x 36 columns, saved to data/clean/student_clean.csv.
