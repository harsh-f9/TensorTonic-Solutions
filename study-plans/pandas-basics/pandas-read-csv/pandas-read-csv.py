import pandas as pd

def create_dataframe(data):
    df=pd.DataFrame(data)
    return {
        "data":df.to_dict("list"),
        "shape": list(df.shape),
        "columns":df.columns.tolist()
    }
