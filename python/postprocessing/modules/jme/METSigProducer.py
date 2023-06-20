import ROOT
import os
import numpy as np
import math
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module



class METSigProducer(Module):

    def __init__(self, JERera, parameters, METCollection="MET", useRecorr=True, calcVariations=False, jetThreshold=15., vetoEtaRegion=(10,10)):
        jetCorrParam = ROOT.JetCorrectorParameters()

        self.pars               = parameters
        self.JERera             = JERera
        self.METCollection      = METCollection
        self.useRecorr          = useRecorr
        self.calcVariations     = calcVariations
        self.jetThreshold       = jetThreshold
        self.JetResolutionFile  = "$CMSSW_BASE/src/JetMETCorrections/Modules/src/JetResolution.cc+"
        self.JERdirectory       = "$CMSSW_BASE/src/PhysicsTools/NanoAODTools/data/jme/"
        self.JetResolutionFile  = os.path.expandvars(self.JetResolutionFile)
        self.vetoEtaRegion      = vetoEtaRegion
        #ROOT.gROOT.ProcessLine('.L '+self.JetResolutionFile)        


    def beginJob(self):
        self.JERdirectory   = os.path.expandvars(self.JERdirectory)
        self.res_pt         = ROOT.JME.JetResolution("%s/%s_PtResolution_AK4PFchs.txt"%(self.JERdirectory, self.JERera))
        self.res_phi        = ROOT.JME.JetResolution("%s/%s_PhiResolution_AK4PFchs.txt"%(self.JERdirectory, self.JERera))
        #self.jer_SF         = ROOT.JME.JetResolutionScaleFactor("%s/%s_SF_AK4PFchs.txt"%(self.JERdirectory, self.JERera))

    def endJob(self):
        del self.res_pt
        del self.res_phi
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("MET_significance", "F")
        if self.calcVariations:
            for var in ['_jesTotalUp', '_jesTotalDown', '_jer', '_jerUp', '_jerDown', '_unclustEnUp', '_unclustEnDown']:
                self.out.branch("MET_significance"+var, "F")
        #self.out.branch("MET_significance_nom", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def getBin(self, abseta):
        etabins = [0.8,1.3,1.9,2.5,100]
        for i, a in enumerate(etabins):
            if abseta < a:
                return int(i)
                break

    def deltaPhi(self, phi1, phi2):
        dphi = phi2-phi1
        if  dphi > math.pi:
            dphi -= 2.0*math.pi
        if dphi <= -math.pi:
            dphi += 2.0*math.pi
        return abs(dphi)

    def deltaR2(self, l1, l2):
        return self.deltaPhi(l1.phi, l2.phi)**2 + (l1.eta - l2.eta)**2

    def deltaR(self, l1, l2):
        return math.sqrt(self.deltaR2(l1,l2))

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets        = Collection(event, "Jet")
        electrons   = Collection(event, "Electron")
        muons       = Collection(event, "Muon")
        photons     = Collection(event, "Photon")
        met         = Object(event, self.METCollection)
        metStd      = Object(event, "MET")
        rho         = getattr(event, "fixedGridRhoFastjetAll")

        ## calculate MET significance for all variants of MET/jets
        if self.useRecorr: variations = ['_nom']
        else: variations = ['']

        if self.calcVariations:
            variations += ['_jesTotalUp', '_jesTotalDown', '_jerUp', '_jer', '_jerDown', '_unclustEnUp', '_unclustEnDown']

        jetPtVar = 'pt' if not self.useRecorr else 'pt_nom'

        for var in variations:
            # no unclustered energy uncertainty for jets
            if var in ['_jesTotalUp', '_jesTotalDown', '_jerUp', '_jerDown', '_jer']:
                jetPtVar = 'pt'+var
            metPtVar = 'pt'+var
            phiVar = 'phi'+var

            # clean against electrons, muons and photons with pt>10 GeV
            sumPtFromJets = 0
            cleanJets = []
            for j in jets:
                clean = True
                for coll in [electrons,muons,photons]:
                    for l in coll:
                        if l.pt > 10 and self.deltaR(j, l) < 0.4:
                            clean = False
                if clean:
                    if not (self.vetoEtaRegion[0] < abs(j.eta) < self.vetoEtaRegion[1]):
                        if getattr(j, jetPtVar) > self.jetThreshold:
                            cleanJets += [j]
                        else:
                            sumPtFromJets += getattr(j, jetPtVar)

            # get the JER
            jet = ROOT.JME.JetParameters()
            for j in cleanJets:
                jet.setJetEta(j.eta).setJetPt(getattr(j, jetPtVar)).setRho(rho)
                j.dpt   = self.res_pt.getResolution(jet)
                j.dphi  = self.res_phi.getResolution(jet)

            cov_xx  = 0
            cov_xy  = 0
            cov_yy  = 0
            i = 0
            for j in cleanJets:
                index       = self.getBin(abs(j.eta))
                jet_index   = 0 if getattr(j, jetPtVar) < 40 else 1 # split into high/low pt jets

                cj = math.cos(j.phi)
                sj = math.sin(j.phi)
                dpt = self.pars[2*index + jet_index] * getattr(j, jetPtVar) * j.dpt
                dph =                                  getattr(j, jetPtVar) * j.dphi

                dpt *= dpt
                dph *= dph

                cov_xx += dpt*cj*cj + dph*sj*sj
                cov_xy += (dpt-dph)*cj*sj
                cov_yy += dph*cj*cj + dpt*sj*sj

                #if var == '_nom':
                #    print getattr(j, jetPtVar)
                #    print cov_xx, cov_xy, cov_yy

                i += 1

            # unclustered energy
            if 'unclust' in var:
                if 'Up' in var:
                    totalSumPt = metStd.sumPt + metStd.sumPtUnclustEnDeltaUp + sumPtFromJets
                else:
                    totalSumPt = metStd.sumPt - metStd.sumPtUnclustEnDeltaUp + sumPtFromJets
            else:
                totalSumPt = metStd.sumPt + sumPtFromJets


            cov_tt = self.pars[10]**2 + self.pars[11]**2*totalSumPt
            cov_xx += cov_tt
            cov_yy += cov_tt

            det = cov_xx*cov_yy - cov_xy*cov_xy

            if det>0:
                ncov_xx =  cov_yy / det
                ncov_xy = -cov_xy / det
                ncov_yy =  cov_xx / det
            else:
                #print cov_xx, cov_yy, cov_xy
                ncov_xx = cov_xx if cov_xx > 0 else 1
                ncov_yy = cov_yy if cov_yy > 0 else 1
                ncov_xy = cov_xy if cov_xy > 0 else 1

            met_pt  = getattr(met, metPtVar)
            met_phi = getattr(met, phiVar)

            met_x = met_pt * math.cos(met_phi)
            met_y = met_pt * math.sin(met_phi)

            MET_sig = met_x*met_x*ncov_xx + 2*met_x*met_y*ncov_xy + met_y*met_y*ncov_yy
            #if var == '_nom':
            #    print "MET pt,x,y", met_pt, met_x, met_y
            #    print MET_sig
            #MET_sig_old = met.significance
            #self.out.fillBranch("MET_significance_nom", float(MET_sig_old))
            if var == '' or var == '_nom':
                self.out.fillBranch("MET_significance", MET_sig)
            else:
                self.out.fillBranch("MET_significance"+var, MET_sig)

        return True

# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed

metSig = lambda : METSigProducer( "Summer16_25nsV1_MC", [1.0,1.0,1.0,1.0,1.0,0.0,0.5] )

