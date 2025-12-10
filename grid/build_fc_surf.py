#!/usr/bin/env python3
import ROOT

pandora_surf = ROOT.TH2F("ph", ";sin^{2}2#theta_{#mu#mu};#Delta m^{2}", 60, 0, 60, 60, 0, 60)
spine_surf = ROOT.TH2F("sh", ";sin^{2}2#theta_{#mu#mu};#Delta m^{2}", 60, 0, 60, 60, 0, 60)
pandora_one_surf = ROOT.TH2F("pho", ";sin^{2}2#theta_{#mu#mu};#Delta m^{2}", 60, 0, 60, 60, 0, 60)
spine_one_surf = ROOT.TH2F("sho", ";sin^{2}2#theta_{#mu#mu};#Delta m^{2}", 60, 0, 60, 60, 0, 60)

pandora_dchi2s = []
spine_dchi2s = []

x = 0
y = 0

for i in range(3600):
  one = ROOT.TFile.Open(f"/exp/icarus/data/users/jlarkin/feld/pandora_fine_final/pandora_fc_{i}_FC.root")

  one_s = ROOT.TFile.Open(f"/exp/icarus/data/users/jlarkin/feld/spine_fine_final/spine_fc_{i}_FC.root")

  one_tree = one.Get("tree")
  dchi2s_one = [entry.chi2_syst - entry.chi2_osc for entry in one_tree]

  one_tree_s = one_s.Get("tree")
  dchi2s_one_s = [entry.chi2_syst - entry.chi2_osc for entry in one_tree_s]

  one.Close()
  one_s.Close()

  critical_one = sorted(dchi2s_one)[int(len(dchi2s_one) * 0.9)]

  critical_one_68 = sorted(dchi2s_one)[int(len(dchi2s_one) * 0.68)]

  critical_one_s = sorted(dchi2s_one_s)[int(len(dchi2s_one_s) * 0.9)]

  critical_one_68_s = sorted(dchi2s_one_s)[int(len(dchi2s_one_s) * 0.68)]

  pandora_surf.SetBinContent(x+1, y+1, critical_one)

  pandora_one_surf.SetBinContent(x+1, y+1, critical_one_68)

  spine_surf.SetBinContent(x+1, y+1, critical_one_s)

  spine_one_surf.SetBinContent(x+1, y+1, critical_one_68_s)

  pandora_h = ROOT.TH1D(f'pandora_{x}_{y}',';#Delta#chi^{2}', 50, 0, max(dchi2s_one))
  spine_h = ROOT.TH1D(f'spine_{x}_{y}',';#Delta#chi^{2}', 50, 0, max(dchi2s_one_s))

  for d in dchi2s_one:
    pandora_h.Fill(d)

  for d in dchi2s_one_s:
    spine_h.Fill(d)

  pandora_dchi2s.append(pandora_h)
  spine_dchi2s.append(spine_h)

  oldy = y
  y = (y + 1) % 60
  x = x + (oldy + 1) // 60


fout = ROOT.TFile("feldman_surf_60x60.root", "RECREATE")
pandora_surf.Write("pandora_90")
spine_surf.Write("spine_90")
pandora_one_surf.Write("pandora_68")
spine_one_surf.Write("spine_68")
for h in pandora_dchi2s:
  h.Write()
for h in spine_dchi2s:
  h.Write()
