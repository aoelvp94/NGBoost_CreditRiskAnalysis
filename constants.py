cols = [
    "SeriousDlqin2yrs",
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

cols_with_missing_indicators=[]
for i in range (len(cols[1:])):
    cols_with_missing_indicators.append(cols[1:][i])
    cols_with_missing_indicators.append([c+"_bool_missing" for c in cols[1:]][i])