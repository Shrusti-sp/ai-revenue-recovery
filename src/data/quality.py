import pandas as pd
def quality_report(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows=[]
    for name, df in frames.items():
        monetary = df.filter(regex="amount|revenue|value", axis=1).select_dtypes("number")
        negative_count = int((monetary < 0).sum().sum()) if not monetary.empty else 0
        rows.append({"table":name,"rows":len(df),"duplicates":int(df.duplicated().sum()),"missing_values":int(df.isna().sum().sum()),"invalid_negative_amounts":negative_count})
    return pd.DataFrame(rows)
