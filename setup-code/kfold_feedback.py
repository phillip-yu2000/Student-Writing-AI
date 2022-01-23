import pandas as pd

from sklearn.model_selection import KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def main():
    """
    Creates K-folds for training data with iterative-stratification,
    described in the following paper:
    Sechidis K., Tsoumakas G., Vlahavas I. (2011) 
    On the Stratification of Multi-Label Data. 
    In: Gunopulos D., Hofmann T., Malerba D., Vazirgiannis M. (eds) 
    Machine Learning and Knowledge Discovery in Databases. ECML PKDD 2011. 
    Lecture Notes in Computer Science, vol 6913. Springer, Berlin, Heidelberg.
    """
    
    K_FOLDS = 5

    df = pd.read_csv("../feedback-prize-2021/train.csv")

    dfx = pd.get_dummies(
        df, columns=["discourse_type"]
        ).groupby(["id"], as_index=False).sum()
    
    cols = [c for c in dfx.columns if c.startswith("discourse_type_") or c == "id" and c != "discourse_type_num"]
    dfx = dfx[cols]

    mskf = MultilabelStratifiedKFold(
        n_splits=K_FOLDS, shuffle=True, random_state=42
        )
    
    labels = [c for c in dfx.columns if c != "id"]
    dfx_labels = dfx[labels]
    dfx["kfold"] = -1

    for fold, (trn_, val_) in enumerate(mskf.split(dfx, dfx_labels)):
        dfx.loc[val_, "kfold"] = fold

    df = df.merge(dfx[["id", "kfold"]], on="id", how="left")
    df.to_csv("../preprocessed/train_folds.csv", index=False)

if __name__ == "__main__":
    main()