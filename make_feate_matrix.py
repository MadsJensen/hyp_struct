import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

plt.style.use("ggplot")

# data_path = "/Users/au194693/projects/hyp_struct/data"
data_path = "/projects/MINDLAB2011_33-MR-high-order-cogn/" +\
            "scratch/Hypnoproj/MJ/data"
os.chdir(data_path)

file_list = glob.glob("*PALS*BA*")
subjects = list(set([f[:4] for f in file_list]))
subjects.sort()

columns_lh = [
    "StructName", "NumVert_lh", "SurfArea_lh", "GrayVol_lh", "ThickAvg_lh",
    "ThickStd_lh", "MeanCurv_lh", "GausCurv_lh", "FoldInd_lh", "CurvInd_lh"
]

columns_rh = [
    "StructName_rh", "NumVert_rh", "SurfArea_rh", "GrayVol_rh", "ThickAvg_rh",
    "ThickStd_rh", "MeanCurv_rh", "GausCurv_rh", "FoldInd_rh", "CurvInd_rh"
]

all_data = pd.DataFrame()

for subject in subjects:
    lh = pd.read_csv(
        "%s.lh.PALS_BA.stats" % subject, skiprows=52, sep='\s+', header=None)
    rh = pd.read_csv(
        "%s.rh.PALS_BA.stats" % subject, skiprows=52, sep='\s+', header=None)
    shss = pd.read_excel(data_path + "/SHSS_GABA-3_mj.xls")
    shss.columns = ["id", "score"]

    lh.columns = columns_lh
    rh.columns = columns_rh

    both = pd.concat([lh, rh], axis=1)
    both = both.drop('StructName_rh', 1)
    both["id"] = subject

    both["shss"] = 0

    both["shss"] = (
        shss.loc[shss["id"] == int(subject)].score.get_values()[0])

    all_data = all_data.append(both, ignore_index=True)


all_data.to_csv(data_path + "/all_data.csv", index=False)