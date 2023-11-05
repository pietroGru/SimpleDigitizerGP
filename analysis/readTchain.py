import ROOT

path = "mkJbs_*.root"

chain = ROOT.TChain("OPT")
print(f"Files loaded: {chain.Add(path)}")
print(f"Entries loaded: {chain.GetEntries()}")