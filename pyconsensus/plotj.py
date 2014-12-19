#!/usr/bin/env python

import numpy as np
import pandas as pd
from pyconsensus import Oracle
import rpy2.robjects as robj
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr

pd.set_option("display.max_rows", 25)
pd.set_option("display.width", 1000)
np.set_printoptions(linewidth=500)

# > M
#         QID1 QID2 QID3 QID4      QID5 QID6 QID7 QID8 QID9     QID10
# Voter 1    1  0.0  1.0    1 0.4498141    0    0    1    1 0.7488008
# Voter 2    0  0.5  0.5   NA 0.4460967    0    0    1    0 0.7488008
# Voter 3    1  0.0  1.0    1 0.4498141    0    0    1    1        NA

# > Scales
#        QID1 QID2 QID3 QID4 QID5 QID6 QID7 QID8 QID9 QID10
# Scaled    0    0    0    0    1    0    0    0    0     1
# Min       0    0    0    0    0    0    0    0    0     0
# Max       1    1    1    1    1    1    1    1    1     1

# > s <- BinaryScales(M)
# > s
#        QID1 QID2 QID3 QID4 QID5 QID6 QID7 QID8 QID9 QID10
# Scaled    0    0    0    0    0    0    0    0    0     0
# Min       0    0    0    0    0    0    0    0    0     0
# Max       1    1    1    1    1    1    1    1    1     1

reports = [[1,  0.0,  1.0,      1, 0.4498141,    0,    0,    1,    1, 0.7488008],
           [0,  0.5,  0.5, np.nan, 0.4460967,    0,    0,    1,    0, 0.7488008],
           [1,  0.0,  1.0,      1, 0.4498141,    0,    0,    1,    1,    np.nan]]

Results = Oracle(votes=reports).consensus()

columns = ["QID1", "QID2", "QID3", "QID4", "QID5", "QID6", "QID7", "QID8", "QID9", "QID10"]
index = ["Reporter 1","Reporter 2","Reporter 3"]

df = pd.DataFrame(Results["filled"], columns=columns, index=index)
df["Var1"] = df.index
mResults = pd.melt(df, id_vars=["Var1"], var_name="Var2")
mResults.value = pd.Categorical(np.round(mResults.value, 4))
mResults.Var1 = pd.Categorical(mResults.Var1)

# Get scores (opacity)
gain_loss = np.matrix(Results["agents"]["voter_bonus"]) - np.matrix(Results["agents"]["old_rep"])
SC = pd.DataFrame(np.hstack((np.matrix(index).T, gain_loss.T)), columns=["Var1", "GainLoss"])

# Format data
DF = pd.merge(mResults, SC)
DF.columns = ("Reporter", "Event", "Outcome", "Scores")

# Build the plot
# (doesn't work, just use rpy2...)
# p1 = ggplot(DF, aes(x="Outcome", y=1, fill="Reporter", alpha="Scores")) + \
#   geom_bar(stat="identity", colour="black") + \
#   geom_text(aes(label="Reporter", vjust=1), position="stack", alpha=1) + \
#   facet_grid("Event", None, scales="fixed")

plotFunc = robj.r("""
    library(ggplot2)
 
    function (DF) {
        p1 <- ggplot(DF,aes(x=Outcome, y=1, fill=Reporter, alpha=Scores)) +
        geom_bar(stat="identity", colour="black") +
        geom_text(aes(label = Reporter, vjust = 1, ymax = 1), position = "stack", alpha=I(1)) +
        xlab("Outcome") +
        ylab("Unscaled Votes") +
        labs(title="Plot of Judgment Space") +
        theme_bw() +
        facet_grid(Event ~ .)

        # Uncomment this line to save to pdf file
        # pdf("plot.pdf", width=8.5, height=11)

        print(p1)
    }
""")

gr = importr('grDevices')
robj.pandas2ri.activate()
testData_R = robj.conversion.py2ri(DF)
plotFunc(testData_R)
raw_input()
gr.dev_off()
