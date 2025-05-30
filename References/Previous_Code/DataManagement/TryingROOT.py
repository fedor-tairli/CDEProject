import ROOT


ROOT.gSystem.Load("$AUGEROFFLINEROOT/lib/libRecEventKG.so")

old_file = ROOT.TFile.Open('ADST_DAT198580_10570.root')
new_file = ROOT.TFile.Open('ADST_DAT198580_10570_new.root', 'recreate')



print(f'Got old_file: {old_file}')
print(f'Got new_file: {new_file}')

recData = old_file.Get('recData')
print(f'Number of Events : {recData.GetEntries()}')


fStations = recData.GetBranch('event.fSDEvent.fStations')
fIsDense  = recData.GetBranch('event.fSDEvent.fStations.fIsDense')

for i in range(recData.GetEntries()):
    recData.GetEntry(i)
    # event = recData.GetLeaf('event.fSDEvent.fStations.fIsDense')
    # for i in range(event.GetLen()):
    #     IsDense = bool(event.GetValue(i))
    #     print(IsDense)
    stations = recData.GetBranch('event.fSDEvent.fStations')

    for leaf in stations.GetListOfLeaves():
        pass
        

new_file.Close()
old_file.Close()


