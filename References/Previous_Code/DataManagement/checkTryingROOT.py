import ROOT

ROOT.gSystem.Load("$AUGEROFFLINEROOT/lib/libRecEventKG.so")

file = ROOT.TFile.Open('ADST_DAT198580_10570_new.root')

keys = file.GetListOfKeys()

for key in keys:
    print(key.GetName())
    print(key.GetClassName())