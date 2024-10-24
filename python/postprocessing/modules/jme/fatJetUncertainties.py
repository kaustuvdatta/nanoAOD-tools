from PhysicsTools.NanoAODTools.postprocessing.modules.jme.JetReCalibrator import JetReCalibrator
#from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer_Run2UL import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.modules.jme.jetSmearer import jetSmearer
from PhysicsTools.NanoAODTools.postprocessing.tools import matchObjectCollection, matchObjectCollectionMultiple
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
import ROOT
import math
import os
import re
import tarfile
import tempfile
import shutil
import numpy as np
import copy
ROOT.PyConfig.IgnoreCommandLineOptions = True


class fatJetUncertaintiesProducer(Module):
    def __init__(
            self,
            era,
            globalTag,
            jesUncertainties=["Total"],
            archive=None,
            jetType="AK8PFPuppi",
            noGroom=False,
            jerTag="",
            jmrVals=[],
            jmsVals=[],
            isData=False,
            applySmearing=True,
            applyAK8JMRSmearing=False,
            useSubJetSmearingBasedGroomedMass = False,
            applyHEMfix=False,
            splitJER=False
    ):
        self.puAlgo = jetType.split('AK8PF')[1]

        self.era = era
        self.noGroom = noGroom
        self.isData = isData
        self.applySmearing = applySmearing if not (isData) else False  # don't smear for data
        self.applyAK8JMRSmearing = applyAK8JMRSmearing if not (isData) else False  # don't smear for data
        self.useSubJetSmearingBasedGroomedMass = useSubJetSmearingBasedGroomedMass if not (isData) else False  # don't smear for data
        
        self.splitJER = splitJER
        if self.splitJER:
            self.splitJERIDs = list(range(6))
        else:
            self.splitJERIDs = [""]  # "empty" ID for the overall JER

        self.jesUncertainties = jesUncertainties
        self.jesUncertaintiesAK4 = jesUncertainties
        
        # smear of jet pT recommended to account for measured difference in JER between data and simulation.
        # jer smearing is, effectively, a multiplicative factor into the .p4() of the AK8 jet, or its subjets as per relevance,
        # this is effectively then a multiplicative constant to the pT and mass of the jet (constant*jet.p4() doesn't touch eta/phi)
        # the same smearing factor can be applied to the AK8 msoftdrop, as is done by default below, but instead JEC corrected
        # subjets (no addl. AK8-based smear factor for mSD in this case) could also be used to calculate the 'more correct' version of the 
        # corrected msoftdrop; however, we only have AK4CHS-based SFs and uncs anyway, so given my use of AK8PUPPI (and thereby PUPPI subjets),
        # just using the JER AK8 smearings for consistency between corrections to pt and mass/msoftdrop by default
        # JMR, etc. smearings are all turned off since reference values for UL aren't available and compute time is wasted/trees are bulkier
        # for no good reason... 

        if jerTag != "":
            self.jerInputFileName = jerTag + "_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = jerTag + "_SF_" + jetType + ".txt"

            if self.useSubJetSmearingBasedGroomedMass: 
                # hard-coding to use same SF/PtRes and Unc inp. file for smearing subjets as per nominal AK8 file for now, 
                # all AK4/8PUPPI files based on AK4CHS, file-names are simply dummies; should instead have AK4PUPPI corrs here, ideally...
                
                self.jerSubjetInputFileName = jerTag + "_PtResolution_" + jetType + ".txt" 
                self.jerSubjetUncertaintyInputFileName = jerTag + "_SF_" + jetType + ".txt"
                
                #self.jerSubjetInputFileName = jerTag + "_PtResolution_AK4PF" + self.puAlgo + ".txt"
                #self.jerSubjetUncertaintyInputFileName = jerTag + "_SF_AK4PF" + self.puAlgo + ".txt"

        else:
            

            self.jerInputFileName = jerTag + "_PtResolution_" + jetType + ".txt"
            self.jerUncertaintyInputFileName = jerTag + "_SF_" + jetType + ".txt"

            print(
                    "WARNING: jerTag is empty!!! Using default UL jer files which might not be the latest",
                    self.jerInputFileName,
                    self.jerUncertaintyInputFileName

                )
            
            
        # Set jet mass resolution variations: https://twiki.cern.ch/twiki/bin/view/CMS/JetWtagging <=========== outdated for Run 2 UL as of 2021-2024, so commented out for now
        if self.applyAK8JMRSmearing:
            self.jmrVals = jmrVals
            if len(self.jmrVals)==0:
                print(
                    "WARNING: jmrVals is empty but mass/msoftdrop smearing requested?!!! Using default values nom/up/down values [1.0, 1.0, 1.0] "
                )
                self.jmrVals = [1.0, 1.0, 1.0]  # nominal, up, down
                ## Use 2017 values for 2018 until 2018 are released
                #if self.era in ["2017", "2018"]:
                #    self.jmrVals = [1.09, 1.14, 1.04]
        else:
            print(
                    "Running for %s; NOTE TO USER: jmrVals is None since self.applyAK8JMRSmearing==False, so no smearing is to be applied to mass/msoftdrop as per jetSmearer." %jetType
                )
            self.jmrVals = None


        print(self.jerUncertaintyInputFileName,self.jmrVals)
        self.jetSmearer = jetSmearer(globalTag, jetType, self.jerInputFileName,
                                     self.jerUncertaintyInputFileName,
                                     self.jmrVals) #add separate jet smearer obj. to smear subjets if possible in future
        if self.useSubJetSmearingBasedGroomedMass: #ie use smearing on indiv. JEC corrected subjets and use their sum.p4().M for mSD nom, instead of overall smear factor for the AK8
            print("Subjet smearer being created")
            self.subjetSmearer = jetSmearer( globalTag, 'AK4PFchs', self.jerSubjetInputFileName,#hard-coding jet type for subjets, no AK4PFPUPPI smear files
                                             self.jerSubjetUncertaintyInputFileName,
                                             self.jmrVals) #add separate jet smearer obj. to smear subjets if possible in future


        if "AK4" in jetType:
            self.jetBranchName = "Jet"
            self.genJetBranchName = "GenJet"
            self.genSubJetBranchName = None
            self.doGroomed = False
        elif "AK8" in jetType:
            self.jetBranchName = "FatJet"
            self.AK4jetBranchName = "Jet"
            self.subJetBranchName = "SubJet"
            self.genJetBranchName = "GenJetAK8"
            self.genSubJetBranchName = "SubGenJetAK8"
            if not(self.noGroom):
                
                self.doGroomed = True # ie, uncorrect, and recorrect subjets as necessary for msoftdrop JER/C corrections, 
                # none of the other fancy pre-UL, PUPPI SD corrections are verifiably available so while keeping them useable below, still
                # decorrelating the rest of the main analyzer/producer below from their effect on msoftdrop_nom as necessary
                
                self.puppiCorrFile = ROOT.TFile.Open(
                    os.environ['CMSSW_BASE'] +
                    "/src/PhysicsTools/NanoAODTools/data/jme/puppiCorr.root")
                self.puppisd_corrGEN = self.puppiCorrFile.Get(
                    "puppiJECcorr_gen")
                self.puppisd_corrRECO_cen = self.puppiCorrFile.Get(
                    "puppiJECcorr_reco_0eta1v3")
                self.puppisd_corrRECO_for = self.puppiCorrFile.Get(
                    "puppiJECcorr_reco_1v3eta2v5")
            else:
                self.doGroomed = False
        else:
            raise ValueError("ERROR: Invalid jet type = '%s'!" % jetType)
        self.rhoBranchName = "fixedGridRhoFastjetAll"
        self.lenVar = "n" + self.jetBranchName
        self.lenVarGenAK8 = "n" + self.genJetBranchName

        # Set jet mass scale variations
        self.jmsVals = jmsVals
        if len(self.jmsVals)==0:
            print(
                "WARNING: jmsVals is empty!!! Using default values of [1.,0.99,1.01] for a default 1 p.c. unc on JMS "
            )
            # not using in my own analysis, currently, but following prescription of: 
            # https://cms-talk.web.cern.ch/t/soft-drop-mass-uncertainty-in-run2/48502/2
            # a la JME DP note from 2023 : http://cds.cern.ch/record/2865845?ln=en 
            self.jmsVals = [1.00, 0.99, 1.01]  # nominal, down, up
            
        # Read jet energy scale (JES) uncertainties

        self.jesInputArchivePath = os.environ['CMSSW_BASE'] + "/src/PhysicsTools/NanoAODTools/data/jme/"
        
        # Text files are tarred, so, need to extract them first into temporary
        # directory (gets deleted during python memory management at script exit)

        self.jesArchive = tarfile.open(
            self.jesInputArchivePath + globalTag +
            ".tgz", "r:gz") if not archive else tarfile.open(
            self.jesInputArchivePath + archive + ".tgz", "r:gz")

        self.jesInputFilePath = tempfile.mkdtemp()
        self.jesArchive.extractall(self.jesInputFilePath)

        if len(jesUncertainties) == 1 and jesUncertainties[0] == "Total":
            print("len(jesUncertainties) == 1 and jesUncertainties[0] == Total")
            self.jesUncertaintyInputFileName = globalTag + "_Uncertainty_" + jetType + ".txt"
            self.jesUncertaintyAK4InputFileName = globalTag + "_Uncertainty_AK4PF" + self.puAlgo + ".txt" #will use AK4PUPPI unc file, which is clone of AK4CHS ones

        elif jesUncertainties[0] == "Merged" and not (self.isData):
            print("jesUncertainties[0] == Merged and not (self.isData)")
            self.jesUncertaintyInputFileName = "RegroupedV2_" + \
                globalTag + "_UncertaintySources_" + jetType + ".txt" #jetType here is AK8PFPUPPI by my def. usage, but inp. files clones of those for AK4CHS for Run 2 UL JES uncs..

            self.jesUncertaintyAK4InputFileName = "RegroupedV2_" + \
                globalTag + "_UncertaintySources_" + "AK4PFchs" + ".txt" # hard-coding since AK4PUPPI unc sources files are clone of AK4CHS ones and not all years have the file named with PUPPI suffix
            
            #replace CHS with self.puAlgo in above if AK4 & / or AK8PUPPI unc sources are separately ever released
            #effectively AK4CHS RegroupedV2 is copied with names AK8PFCHS and AK8PFPUPPI for now
            #by JME, so instead of adding another copy of the same tagged "AK4PFPUPPi", hard-coding for CHS now
            #relevant when recorrecting subjets with AK4 nom JECs and uncs (albeit the ones from CHS applied to 
            #jet types all), doing this for consistency instead of reusing above global var already used for AK8s
                

        else:
            
            self.jesUncertaintyInputFileName = globalTag + "_UncertaintySources_" + jetType + ".txt"
            
            self.jesUncertaintyAK4InputFileName = globalTag + "_UncertaintySources_AK4PF" + self.puAlgo + ".txt"

        # read all uncertainty source names from the loaded file
        if jesUncertainties[0] in ["All", "Merged"]:
            
            print(self.jesInputFilePath + '/' + self.jesUncertaintyInputFileName)

            with open(self.jesInputFilePath + '/' + self.jesUncertaintyInputFileName) as f:
                lines = f.read().split("\n")
                sources = [
                    x for x in lines if x.startswith("[") and x.endswith("]")
                ]
                sources = [x[1:-1] for x in sources]
                self.jesUncertainties = sources

            
            print(self.jesInputFilePath + '/' +  self.jesUncertaintyAK4InputFileName, self.puAlgo)
            print(self.jesInputFilePath + '/' +  self.jesUncertaintyInputFileName, self.puAlgo)

            with open(self.jesInputFilePath + '/' +  self.jesUncertaintyAK4InputFileName) as f:
                lines = f.read().split("\n")
                sources = [
                    x for x in lines if x.startswith("[") and x.endswith("]")
                ]
                sources = [x[1:-1] for x in sources]
                self.jesUncertaintiesAK4 = sources #need to vary subjet kinematics simultaneously with AK8 jet when running unc. variations

        if applyHEMfix:
            self.jesUncertainties.append("HEMIssue")
            self.jesUncertaintiesAK4.append("HEMIssue")

        print("AK8 recalibrator creation", self.jesInputFilePath)
        self.jetReCalibrator = JetReCalibrator(
            globalTag,
            jetType,
            True,
            self.jesInputFilePath,
            calculateSeparateCorrections=False,
            calculateType1METCorrection=False)

        
        # not hard-coding jetType to be for AK4PFCHS or AK8PFPUPPI etc., here like before, 
        # since JECs derived separately for all types, apparently, fom twiki:https://twiki.cern.ch/twiki/bin/view/CMS/JECDataMC#Jet_Energy_Corrections_in_Run2)
        # only uncs. derived from AK4PFCHS not JECs themselves
        print('AK4PF'+self.puAlgo + " (subjet JEC) recalibrator creation", self.jesInputFilePath)

        self.subjetReCalibrator = JetReCalibrator(
            globalTag,
            'AK4PF'+self.puAlgo,                 
            True,
            self.jesInputFilePath,
            calculateSeparateCorrections=False,
            calculateType1METCorrection=False)

        # load libraries for accessing JES scale factors and uncertainties
        # from txt files
        for library in [
                "libCondFormatsJetMETObjects", "libPhysicsToolsNanoAODTools"
        ]:
            if library not in ROOT.gSystem.GetLibraries():
                print("Load Library '%s'" % library.replace("lib", ""))
                ROOT.gSystem.Load(library)

    def getJERsplitID(self, pt, eta):
        if not self.splitJER:
            return ""
        if abs(eta) < 1.93:
            return 0
        elif abs(eta) < 2.5:
            return 1
        elif abs(eta) < 3:
            if pt < 50:
                return 2
            else:
                return 3
        else:
            if pt < 50:
                return 4
            else:
                return 5

    def beginJob(self):

        print("Loading jet energy scale (JES) uncertainties from file '%s'" %
              os.path.join(self.jesInputFilePath,
                           self.jesUncertaintyInputFileName))
        print("Loading AK4 jet energy scale (JES) uncertainties from file '%s' to calculate JEC correction and uncs for subjets" %
              os.path.join(self.jesInputFilePath,
                           self.jesUncertaintyAK4InputFileName))        

        self.jesUncertainty = {}
        self.jesUncertaintyAK4 = {}
        
        for jesUncertainty in self.jesUncertainties:

            jesUncertainty_label = jesUncertainty
            if jesUncertainty == 'Total' and (len(self.jesUncertainties) == 1 or (len(self.jesUncertainties) == 2 and 'HEMIssue' in self.jesUncertainties)):
                jesUncertainty_label = ''
                print("jesUncertainty_label", jesUncertainty_label)


            if jesUncertainty != "HEMIssue":


                print(jesUncertainty, "in jesUnc not HEMIssue if statement in fatjetUncertainties script",
                      os.path.join(self.jesInputFilePath,self.jesUncertaintyInputFileName),
                      jesUncertainty_label)

                pars = ROOT.JetCorrectorParameters(
                            os.path.join(self.jesInputFilePath,
                                         self.jesUncertaintyInputFileName),
                            jesUncertainty_label)
                #print("pars",pars)
                parsAK4 = ROOT.JetCorrectorParameters(
                            os.path.join(self.jesInputFilePath,
                                         self.jesUncertaintyAK4InputFileName),
                            jesUncertainty_label)

                #print("parsAK4", parsAK4)

                self.jesUncertainty[jesUncertainty] = ROOT.JetCorrectionUncertainty(pars)
                self.jesUncertaintyAK4[jesUncertainty] = ROOT.JetCorrectionUncertainty(parsAK4)

        if not (self.isData):
            print("Initialise fatjet smearer object")

            self.jetSmearer.beginJob()
            if self.useSubJetSmearingBasedGroomedMass:
                print("Initialise subjet smearer too, for subjet-JER-corr-included msoftdrop")
                self.subjetSmearer.beginJob()

    def endJob(self):
        if not (self.isData):
            self.jetSmearer.endJob()
            if self.useSubJetSmearingBasedGroomedMass:
                print("Ending subjet smearing step used to get subjet-JER-corr-included msoftdrop")
                self.subjetSmearer.endJob()

        shutil.rmtree(self.jesInputFilePath)

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        
        self.out.branch("%s_pt_raw" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_pt_nom" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_mass_raw" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_mass_nom" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_corr_JEC" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_corr_JER" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        self.out.branch("%s_corr_JMS" % self.jetBranchName, "F",
                        lenVar=self.lenVar)

        
        if self.doGroomed:
            self.out.branch("%s_msoftdrop_raw" % self.jetBranchName,"F",
                            lenVar=self.lenVar)
            self.out.branch("%s_msoftdrop_nom" % self.jetBranchName,"F",
                            lenVar=self.lenVar)
            self.out.branch("%s_msoftdrop_corr_subjetJEC" % self.jetBranchName,"F",
                            lenVar=self.lenVar)
            self.out.branch("%s_msoftdrop_nom_PUPPICorred" % self.jetBranchName,"F",
                            lenVar=self.lenVar)
            if not(self.isData):
                self.out.branch("%s_msoftdrop_corr_JMS" % self.jetBranchName,"F",
                                lenVar=self.lenVar)
                self.out.branch("%s_msoftdrop_corr_PUPPI" % self.jetBranchName,"F",
                                lenVar=self.lenVar)
                #print("%s_msoftdrop_two_subjets" % self.genJetBranchName)
                #self.out.branch("%s_msoftdrop_two_subjets" % self.genJetBranchName, "F",
                #                lenVar=self.lenVarGenAK8)
                #self.out.branch("%s_msoftdrop_three_subjets" % self.genJetBranchName, "F",
                #                lenVar=self.lenVarGenAK8)
                #self.out.branch("%s_msoftdrop_four_subjets" % self.genJetBranchName, "F",
                #                lenVar=self.lenVarGenAK8)
                #self.out.branch("%s_msoftdrop_five_subjets" % self.genJetBranchName, "F",
                #                lenVar=self.lenVarGenAK8)

                if self.applyAK8JMRSmearing:

                    self.out.branch("%s_msoftdrop_corr_JMR" % self.jetBranchName,"F",
                                    lenVar=self.lenVar)
                    self.out.branch("%s_corr_JMR" % self.jetBranchName,"F",
                                    lenVar=self.lenVar)


                
        if not (self.isData):
            #self.out.branch("%s_msoftdrop_tau21DDT_nom" % self.jetBranchName,
            #                "F",
            #                lenVar=self.lenVar)
            for shift in ["Up", "Down"]:
                for jerID in self.splitJERIDs:
                    self.out.branch("%s_pt_jer%s%s" %
                                    (self.jetBranchName, jerID, shift),
                                    "F",
                                    lenVar=self.lenVar)
                    self.out.branch("%s_mass_jer%s%s" %
                                    (self.jetBranchName, jerID, shift),
                                    "F",
                                    lenVar=self.lenVar)

                if self.applyAK8JMRSmearing:
                    self.out.branch("%s_mass_jmr%s" % (self.jetBranchName, shift),
                                    "F",
                                    lenVar=self.lenVar)

                self.out.branch("%s_mass_jms%s" % (self.jetBranchName, shift),
                                "F",
                                lenVar=self.lenVar)

                if self.doGroomed:
                    for jerID in self.splitJERIDs:
                        self.out.branch("%s_msoftdrop_jer%s%s" %
                                        (self.jetBranchName, jerID, shift),
                                        "F",
                                        lenVar=self.lenVar)
                        #self.out.branch("%s_msoftdrop_tau21DDT_jer%s%s" %
                        #                (self.jetBranchName, jerID, shift),
                        #                "F",
                        #                lenVar=self.lenVar)
                    if self.applyAK8JMRSmearing:
                        self.out.branch("%s_msoftdrop_jmr%s" %
                                        (self.jetBranchName, shift),
                                        "F",
                                        lenVar=self.lenVar)
                    
                    self.out.branch("%s_msoftdrop_jms%s" %
                                    (self.jetBranchName, shift),
                                    "F",
                                    lenVar=self.lenVar)
                    
                    #self.out.branch("%s_msoftdrop_tau21DDT_jmr%s" %
                    #                (self.jetBranchName, shift),
                    #                "F",
                    #                lenVar=self.lenVar)
                    #self.out.branch("%s_msoftdrop_tau21DDT_jms%s" %
                    #                (self.jetBranchName, shift),
                    #                "F",
                    #                lenVar=self.lenVar)

                for jesUncertainty in self.jesUncertainties:
                    self.out.branch(
                        "%s_pt_jes%s%s" %
                        (self.jetBranchName, jesUncertainty, shift),
                        "F",
                        lenVar=self.lenVar)
                    self.out.branch(
                        "%s_mass_jes%s%s" %
                        (self.jetBranchName, jesUncertainty, shift),
                        "F",
                        lenVar=self.lenVar)
                    if self.doGroomed:
                        self.out.branch(
                            "%s_msoftdrop_jes%s%s" %
                            (self.jetBranchName, jesUncertainty, shift),
                            "F",
                            lenVar=self.lenVar)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""
        jets = Collection(event, self.jetBranchName)

        if not(self.isData):
            genJets = Collection(event, self.genJetBranchName)
            self.jetSmearer.setSeed(event)

        if self.doGroomed:
            subJets = Collection(event, self.subJetBranchName)
            AK4Jets = Collection(event, self.AK4jetBranchName)
            if not(self.isData):
                genSubJets = Collection(event, self.genSubJetBranchName)
                
        
        jets_pt_raw = []
        jets_pt_nom = []
        jets_mass_raw = []
        jets_mass_nom = []

        jets_corr_JEC = []
        jets_corr_JER = []
        jets_corr_JMS = []
        

        jets_pt_jerUp = {}
        jets_pt_jerDown = {}
        jets_pt_jesUp = {}
        jets_pt_jesDown = {}

        jets_mass_jerUp = {}
        jets_mass_jerDown = {}
        jets_mass_jesUp = {}
        jets_mass_jesDown = {}
        jets_mass_jmsUp = []
        jets_mass_jmsDown = []

        if self.applyAK8JMRSmearing:
            jets_mass_jmrUp = []
            jets_mass_jmrDown = []
            jets_corr_JMR = []

        for jerID in self.splitJERIDs:
            jets_pt_jerUp[jerID] = []
            jets_pt_jerDown[jerID] = []
            jets_mass_jerUp[jerID] = []
            jets_mass_jerDown[jerID] = []

        for jesUncertainty in self.jesUncertainties:
            jets_pt_jesUp[jesUncertainty] = []
            jets_pt_jesDown[jesUncertainty] = []
            jets_mass_jesUp[jesUncertainty] = []
            jets_mass_jesDown[jesUncertainty] = []

        if self.doGroomed:
            jets_msdcorr_raw = []
            jets_msdcorr_nom = []
            jets_msdcorr_nom_PUPPICorred = []
            jets_msdcorr_subjetJEC = []

            if self.applyAK8JMRSmearing:
                jets_msdcorr_corr_JMR = []
                jets_msdcorr_jmrUp = []
                jets_msdcorr_jmrDown = []
            
            #genAK8_msd_from_two_subjets = []
            #genAK8_msd_from_three_subjets = []
            #genAK8_msd_from_four_subjets = []
            #genAK8_msd_from_five_subjets = []

            jets_msdcorr_corr_JMS = []
            jets_msdcorr_corr_PUPPI = []
            jets_msdcorr_corr_subjetJEC = []
            jets_msdcorr_jerUp = {}
            jets_msdcorr_jerDown = {}
            jets_msdcorr_jesUp = {}
            jets_msdcorr_jesDown = {}
            jets_msdcorr_jmsUp = []
            jets_msdcorr_jmsDown = []
            #jets_msdcorr_tau21DDT_nom = []
            #jets_msdcorr_tau21DDT_jerUp = {}
            #jets_msdcorr_tau21DDT_jerDown = {}
            #jets_msdcorr_tau21DDT_jmrUp = []
            #jets_msdcorr_tau21DDT_jmrDown = []
            #jets_msdcorr_tau21DDT_jmsUp = []
            #jets_msdcorr_tau21DDT_jmsDown = []
            for jerID in self.splitJERIDs:
                jets_msdcorr_jerUp[jerID] = []
                jets_msdcorr_jerDown[jerID] = []
                #jets_msdcorr_tau21DDT_jerUp[jerID] = []
                #jets_msdcorr_tau21DDT_jerDown[jerID] = []
            for jesUncertainty in self.jesUncertainties:
                jets_msdcorr_jesUp[jesUncertainty] = []
                jets_msdcorr_jesDown[jesUncertainty] = []

        rho = getattr(event, self.rhoBranchName)

        def resolution_matching(jet, genjet, res_factor=3,smearer=self.jetSmearer):# extend to subjets for JER smearing of subjets for mSD in future
            '''Helper function to match to gen based on pt difference'''
            params = ROOT.PyJetParametersWrapper()
            params.setJetEta(jet.eta)
            params.setJetPt(jet.pt)
            params.setRho(rho)

            resolution = smearer.jer.getResolution(params)

            ##################
            # 3 * sigma * reco pT
            ##################
            return abs(jet.pt - genjet.pt) < res_factor * resolution * jet.pt

        # match reconstructed jets to generator level ones
        # (needed to evaluate JER scale factors and uncertainties)
        if not(self.isData):
            pairs = matchObjectCollection(jets, 
                                          genJets, 
                                          dRmax=0.4, 
                                          presel=resolution_matching)
            #pairs_subjets = matchObjectCollection(subJets, 
            #                                      AK4Jets, 
            #                                      dRmax=0.6,  # keeping loose and hoping closest satisfies less than at least 0.2
            #                                      )#presel=resolution_matching)
            

            genSubJetMatcher = matchObjectCollectionMultiple(genJets,
                                                             genSubJets,
                                                             dRmax=0.8)
            recoSubJetMatcher = matchObjectCollectionMultiple(jets,
                                                              subJets,
                                                              dRmax=0.8)

        for jet in jets:

            # jet pt and mass corrections by latest JECS
            jet_pt = jet.pt
            jet_mass = jet.mass

            if hasattr(jet, "rawFactor"):
                jet_rawpt = jet_pt * (1 - jet.rawFactor)
                jet_rawmass = jet_mass * (1 - jet.rawFactor)
            else:
                jet_rawpt = -1.0 * jet_pt  # If factor not present quantity will be saved as -1; KD: what cases would they be unavailable in!?
                jet_rawmass = -1.0 * jet_mass  

            (jet_pt, jet_mass) = self.jetReCalibrator.correct(jet, rho)
            jet.pt = jet_pt
            jet.mass = jet_mass
            jets_pt_raw.append(jet_rawpt)
            jets_mass_raw.append(jet_rawmass)
            jets_corr_JEC.append(jet_pt / jet_rawpt)

            if not(self.isData):
                genJet = pairs[jet]

            # evaluate JER scale factors and uncertainties
            # (cf. https://twiki.cern.ch/twiki/bin/view/CMS/JetResolution and https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyResolution )
            if not(self.isData):
                (jet_pt_jerNomVal, jet_pt_jerUpVal,
                 jet_pt_jerDownVal) = self.jetSmearer.getSmearValsPt(
                     jet, genJet, rho)
            else:
                # set values to 1 for data so that jet_pt_nom is not smeared
                (jet_pt_jerNomVal, jet_pt_jerUpVal, jet_pt_jerDownVal) = (1, 1, 1)
            jets_corr_JER.append(jet_pt_jerNomVal)

            jet_pt_nom = jet_pt_jerNomVal * jet_pt if self.applySmearing else jet_pt 

            if jet_pt_nom < 0.0:
                jet_pt_nom *= -1.0
            jets_pt_nom.append(jet_pt_nom)

            # Evaluate JMS and JMR scale factors and uncertainties
            jmsNomVal, jmsDownVal, jmsUpVal = self.jmsVals if not(self.isData) else (1, 1, 1)
            jets_corr_JMS.append(jmsNomVal)

            if self.applyAK8JMRSmearing:
                if not(self.isData):
                    (jet_mass_jmrNomVal, jet_mass_jmrUpVal,
                     jet_mass_jmrDownVal) = self.jetSmearer.getSmearValsM(
                         jet, genJet)
                else:
                    # set values to 1 for data so that jet_mass_nom is not smeared
                    (jet_mass_jmrNomVal, jet_mass_jmrUpVal,jet_mass_jmrDownVal) = (1, 1, 1)
                
                jets_corr_JMR.append(jet_mass_jmrNomVal)
                jet_mass_nom = jet_pt_jerNomVal * jet_mass_jmrNomVal * jmsNomVal * jet_mass if self.applySmearing else jmsNomVal * jet_mass
            else:
                jet_mass_nom = jet_pt_jerNomVal * jmsNomVal * jet_mass if self.applySmearing else jmsNomVal * jet_mass

            if jet_mass_nom < 0.0:
                jet_mass_nom *= -1.0
            jets_mass_nom.append(jet_mass_nom)

            if not(self.isData):
                jet_pt_jerUp = {
                    jerID: jet_pt_nom
                    for jerID in self.splitJERIDs
                }
                jet_pt_jerDown = {
                    jerID: jet_pt_nom
                    for jerID in self.splitJERIDs
                }
                jet_mass_jerUp = {
                    jerID: jet_mass_nom
                    for jerID in self.splitJERIDs
                }
                jet_mass_jerDown = {
                    jerID: jet_mass_nom
                    for jerID in self.splitJERIDs
                }
                thisJERID = self.getJERsplitID(jet_pt_nom, jet.eta)



                jet_pt_jerUp[thisJERID] = jet_pt_jerUpVal * jet_pt
                jet_pt_jerDown[thisJERID] = jet_pt_jerDownVal * jet_pt
                jet_mass_jerUp[thisJERID] = jet_pt_jerUpVal * (1. if not(self.applyAK8JMRSmearing) else jet_mass_jmrNomVal) * jmsNomVal * jet_mass
                jet_mass_jerDown[thisJERID] = jet_pt_jerDownVal * (1. if not(self.applyAK8JMRSmearing) else jet_mass_jmrNomVal) * jmsNomVal * jet_mass

                for jerID in self.splitJERIDs:
                    jets_pt_jerUp[jerID].append(jet_pt_jerUp[jerID])
                    jets_pt_jerDown[jerID].append(jet_pt_jerDown[jerID])
                    jets_mass_jerUp[jerID].append(jet_mass_jerUp[jerID])
                    jets_mass_jerDown[jerID].append(jet_mass_jerDown[jerID])

                if self.applyAK8JMRSmearing:
                    jets_mass_jmrUp.append(jet_pt_jerNomVal * jet_mass_jmrUpVal *
                                           jmsNomVal * jet_mass)
                    jets_mass_jmrDown.append(jet_pt_jerNomVal *
                                             jet_mass_jmrDownVal * jmsNomVal *
                                             jet_mass)
                
                jets_mass_jmsUp.append(jet_pt_jerNomVal * (1. if not(self.applyAK8JMRSmearing) else jet_mass_jmrNomVal) * jmsUpVal * jet_mass)
                jets_mass_jmsDown.append(jet_pt_jerNomVal * (1. if not(self.applyAK8JMRSmearing) else jet_mass_jmrNomVal) * jmsDownVal * jet_mass)

            if self.doGroomed:

                if not(self.isData):

                    genGroomedSubJets = genSubJetMatcher[genJet] if not(genJet==None) else None

 
                    genGroomedJet_base = genGroomedSubJets[0].p4() if not(genGroomedSubJets==None) and len(genGroomedSubJets) >= 1 else 0.

                    genGroomedJet = genGroomedSubJets[0].p4() + genGroomedSubJets[1].p4() if genGroomedSubJets is not None and len(genGroomedSubJets) >= 2 else None
                    """                    
                    if genGroomedSubJets!=None and len(genGroomedSubJets) >= 3: 
                        genGroomedJet3 = genGroomedJet + genGroomedSubJets[2].p4() if not(genGroomedJet==None) else None
                    else:
                        genGroomedJet3 = None
                    if genGroomedSubJets!=None and len(genGroomedSubJets) >= 4: 
                        genGroomedJet4 = genGroomedJet3 + genGroomedSubJets[3].p4() if not(genGroomedJet3==None) else None
                    else:
                        genGroomedJet4 = None
                    if genGroomedSubJets!=None and len(genGroomedSubJets) >= 5: 
                        genGroomedJet5 = genGroomedJet4 + genGroomedSubJets[4].p4() if not(genGroomedJet4==None) else None
                    else:
                        genGroomedJet5 = None

                    genAK8_msd_from_two_subjets.append(genGroomedJet.M() if not(genGroomedJet==None) else 0.)#genGroomedJet_base )

                    if not(genGroomedJet==None):
                        genAK8_msd_from_three_subjets.append(genGroomedJet3.M() if not(genGroomedJet3==None) else 0.)#genGroomedJet.M() )
                    else: 
                        genAK8_msd_from_three_subjets.append(0.)

                    if not(genGroomedJet3==None):
                        genAK8_msd_from_four_subjets.append(genGroomedJet4.M() if not(genGroomedJet4==None)  else 0.)#genGroomedJet3.M() )
                    else: 
                        genAK8_msd_from_four_subjets.append(0.)

                    if not(genGroomedJet4==None):
                        genAK8_msd_from_five_subjets.append(genGroomedJet5.M() if not(genGroomedJet5==None)  else 0.)#genGroomedJet4.M() )
                    else: 
                        genAK8_msd_from_five_subjets.append(0.)

                    """
                else:
                    genGroomedSubJets = None
                    genGroomedJet = None

                if jet.subJetIdx1 >= 0 and jet.subJetIdx2 >= 0:

                    ############for x in subJets:
                    #########AK4 = pairs_subjets[x]


                    #incorporating change proposed as per: https://github.com/cms-nanoAOD/nanoAOD-tools/issues/280 where one undoes 
                    #application of JECs for subjets to get the correctly uncorrected msoftdrop_raw, AK8 JEC (+ syst. variations ) 
                    #applied thereof to msoftdrop_raw at analysis level
                    
                    groomedP4_raw = (subJets[jet.subJetIdx1].p4() * (1. - subJets[jet.subJetIdx1].rawFactor)) + (subJets[jet.subJetIdx2].p4() * (1. - subJets[jet.subJetIdx2].rawFactor))

                    # use jet catchment area of nearest AK4... seems incorrect a priori, not sure how to proceed, scratch that, use 'default' AK4 area a la naive pi*R^2 after chat with A.M.

                    subJets[jet.subJetIdx1].area = ROOT.TMath.Pi()*( (0.4)**2 )#pairs_subjets[subJets[jet.subJetIdx1]].area
                    subJets[jet.subJetIdx2].area = ROOT.TMath.Pi()*( (0.4)**2 )#pairs_subjets[subJets[jet.subJetIdx2]].area

                    # get JEC corrected subjets
                    subJets1_p4 = self.subjetReCalibrator.correct_return_p4(subJets[jet.subJetIdx1], rho)
                    subJets2_p4 = self.subjetReCalibrator.correct_return_p4(subJets[jet.subJetIdx2], rho)

                    groomedP4_reJECed_SJs = subJets1_p4 + subJets2_p4
                    
                else:
                    groomedP4_raw = None
                    groomedP4_reJECed_SJs = None

                jet_msdcorr_raw = groomedP4_raw.M() if groomedP4_raw is not None else 0.0

                jet_msdcorr_subjetJEC = groomedP4_reJECed_SJs.M() if groomedP4_reJECed_SJs is not None else 0.0

                # now apply the mass correction to the raw value
                if jet_msdcorr_raw < 0.0:
                    jet_msdcorr_raw *= -1.0
                
                if jet_msdcorr_subjetJEC < 0.0:
                    jet_msdcorr_subjetJEC *= -1.0



                # raw value stored from SJ1+2's 4-mom. after 'uncorrecting JECs', without any further corrections applied
                jets_msdcorr_raw.append(jet_msdcorr_raw)
                jets_msdcorr_subjetJEC.append(jet_msdcorr_subjetJEC)
                jets_msdcorr_corr_subjetJEC.append( jet_msdcorr_subjetJEC / jet_msdcorr_raw if jet_msdcorr_raw!=0.0 else -1. )  #store overall corr. factor for groomedP4 mass after correcting subjet 4-mom. with AK4 JECs



                # LC: Apply PUPPI SD mass correction https://github.com/cms-jet/PuppiSoftdropMassCorr/
                ## KD: not used since repo is not updated for UL Run 2, keeping code snippet in here
                ## so that we store the outdated correction nonetheless, to not make changes further downstream

                puppisd_genCorr = self.puppisd_corrGEN.Eval(jet.pt)
                if abs(jet.eta) <= 1.3:
                    puppisd_recoCorr = self.puppisd_corrRECO_cen.Eval(jet.pt)
                else:
                    puppisd_recoCorr = self.puppisd_corrRECO_for.Eval(jet.pt)

                puppisd_total = puppisd_genCorr * puppisd_recoCorr
                jets_msdcorr_corr_PUPPI.append(puppisd_total)

                if groomedP4_raw is not None:
                    groomedP4_puppiCorred = copy.deepcopy(groomedP4_raw)

                    groomedP4_puppiCorred.SetPtEtaPhiM(groomedP4_puppiCorred.Pt(), groomedP4_puppiCorred.Eta(),
                                                       groomedP4_puppiCorred.Phi(),
                                                       groomedP4_puppiCorred.M() * puppisd_total)
                else:
                    groomedP4_puppiCorred = None


                jet_msdcorr_raw_puppiCorred = groomedP4_puppiCorred.M() if groomedP4_puppiCorred is not None else 0.0
                if jet_msdcorr_raw_puppiCorred < 0.0:
                    jet_msdcorr_raw_puppiCorred *= -1.0

                # Evaluate JMS and JMR scale factors and uncertainties
                jets_msdcorr_corr_JMS.append(jmsNomVal) #1 by default

                if self.applyAK8JMRSmearing:
                    if not(self.isData):
                        # prety sure what this block is doing is wrong, outdated corr and otherwise, the raw p4 mass was being corrected with some puppi corr
                        # then, some jmr smearing was being calculated with that P4 vs. the gen groomed jet ( variables such as p4's here are renamed 
                        # for the sake of at least some mnemonic association of variable names to physical objects);
                        # firstly the gen groomed jet isn't dR or pt resol. 
                        # matched to the reco groomed p4, so smearing is hard to justify and inconsistent with how AK4's in jetmetUncertainties or AK8's above
                        # are being matched prior to smearing, whether that be for the energy or simply the mass (JMR smearing is off by default for my analysis though)
                        # so figuring out what is the best, least incorrect fix for this to workaround outdated puppi SD corrs isn't something I will investigate now
                        # given that they aren't recommended corrs anymore either 

                        (jet_msdcorr_jmrNomVal, jet_msdcorr_jmrUpVal,
                         jet_msdcorr_jmrDownVal) = ( self.jetSmearer.getSmearValsM(groomedP4_puppiCorred,
                                                     genGroomedJet) if groomedP4_puppiCorred is not None
                                                     and genGroomedJet is not None else (0., 0., 0.))
                    else:
                        (jet_msdcorr_jmrNomVal, jet_msdcorr_jmrUpVal,
                         jet_msdcorr_jmrDownVal) = (1, 1, 1)
                    
                    jets_msdcorr_corr_JMR.append(jet_msdcorr_jmrNomVal)
                    # pretty sure in below, where the puppi corrected 'raw' mSD was being used, they should've been using the real raw mSD
                    # a la:
                    # jet_msdcorr_nom = jet_pt_jerNomVal * jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_raw if self.applySmearing else jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_raw
                    # or more correctly even, maybe, the msdcorr after JECs reappleid to subjets:
                    # jet_msdcorr_nom = jet_pt_jerNomVal * jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_subjetJEC if self.applySmearing else jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_subjetJEC
                    # but keeping as was, modulo name change of variables, for consistency between versions even if quantity not used downstream
                    jet_msdcorr_nom = jet_pt_jerNomVal * jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_raw_puppiCorred if self.applySmearing else jet_msdcorr_jmrNomVal * jmsNomVal * jet_msdcorr_raw_puppiCorred
                    jet_msdcorr_nom_PUPPICorred = jet_msdcorr_nom
                else:
                    ##################################
                    # my default msoftdrop, with JEC corrected subjets, and smear factor based on AK8 energy instead of subjet smear factors
                    # but not psosible since subjet areas not available in nano... using overall AK8 jec instead of subjet JECs
                    jet_msdcorr_nom = jet_pt_jerNomVal * jmsNomVal * jet_msdcorr_subjetJEC if self.applySmearing else jmsNomVal * jet_msdcorr_subjetJEC
                    #jet_msdcorr_nom = jet_pt_jerNomVal * jmsNomVal * (jet_pt/jet_rawpt) * jet_msdcorr_raw if self.applySmearing else jmsNomVal * (jet_pt/jet_rawpt) * jet_msdcorr_raw 
                    jet_msdcorr_nom_PUPPICorred = jet_pt_jerNomVal * jmsNomVal * (jet_pt/jet_rawpt) * jet_msdcorr_raw_puppiCorred if self.applySmearing else jmsNomVal * (jet_pt/jet_rawpt) * jet_msdcorr_raw_puppiCorred 

                    ##################################
                
                # store the nominal mass value, ideally with JEC corrected subjets, but simply sticking to JEC factor for AK8 since subjet catchment areas unavailable
                jets_msdcorr_nom.append(jet_msdcorr_nom)
                jets_msdcorr_nom_PUPPICorred.append(jet_msdcorr_nom_PUPPICorred)

                if not self.isData:
                    jet_msdcorr_jerUp = {
                        jerID: jet_msdcorr_nom
                        for jerID in self.splitJERIDs
                    }
                    jet_msdcorr_jerDown = {
                        jerID: jet_msdcorr_nom
                        for jerID in self.splitJERIDs
                    }
                    thisJERID = self.getJERsplitID(jet_pt_nom, jet.eta)
                    jet_msdcorr_jerUp[thisJERID] = jet_pt_jerUpVal * (1. if not(self.applyAK8JMRSmearing) else jet_msdcorr_jmrNomVal) * jmsNomVal * jet_msdcorr_subjetJEC #* (jet_pt/jet_rawpt) * jet_msdcorr_raw  #* jet_msdcorr_raw
                    jet_msdcorr_jerDown[thisJERID] = jet_pt_jerDownVal * (1. if not(self.applyAK8JMRSmearing) else jet_msdcorr_jmrNomVal) * jmsNomVal * jet_msdcorr_subjetJEC #* (jet_pt/jet_rawpt) * jet_msdcorr_raw  #* jet_msdcorr_raw

                    for jerID in self.splitJERIDs:
                        jets_msdcorr_jerUp[jerID].append(jet_msdcorr_jerUp[jerID])
                        jets_msdcorr_jerDown[jerID].append(jet_msdcorr_jerDown[jerID])

                    if self.applyAK8JMRSmearing:
                        jets_msdcorr_jmrUp.append(
                            jet_pt_jerNomVal * jet_msdcorr_jmrUpVal * jmsNomVal *
                            jet_msdcorr_raw)
                        jets_msdcorr_jmrDown.append(
                            jet_pt_jerNomVal * jet_msdcorr_jmrDownVal * jmsNomVal *
                            jet_msdcorr_raw)

                    jets_msdcorr_jmsUp.append(
                        jet_pt_jerNomVal * (1. if not(self.applyAK8JMRSmearing) else jet_msdcorr_jmrNomVal) * jmsUpVal * jet_msdcorr_subjetJEC) #jet_msdcorr_raw) #* (jet_pt/jet_rawpt) * jet_msdcorr_raw)
                    jets_msdcorr_jmsDown.append(
                        jet_pt_jerNomVal * (1. if not(self.applyAK8JMRSmearing) else jet_msdcorr_jmrNomVal) * jmsDownVal * jet_msdcorr_subjetJEC) #jet_msdcorr_raw) #* (jet_pt/jet_rawpt) * jet_msdcorr_raw)

                    # Also evaluated JMS&JMR SD corr in tau21DDT region: https://twiki.cern.ch/twiki/bin/viewauth/CMS/JetWtagging#tau21DDT_0_43
                    #if self.era in ["2016"]:
                    #    jmstau21DDTNomVal = 1.014
                    #    jmstau21DDTDownVal = 1.007
                    #    jmstau21DDTUpVal = 1.021
                    #    self.jetSmearer.jmr_vals = [1.086, 1.176, 0.996]
                    #elif self.era in ["2017"]:
                    #    jmstau21DDTNomVal = 0.983
                    #    jmstau21DDTDownVal = 0.976
                    #    jmstau21DDTUpVal = 0.99
                    #    self.jetSmearer.jmr_vals = [1.080, 1.161, 0.999]
                    #elif self.era in ["2018"]:
                    #    jmstau21DDTNomVal = 1.000  # tau21DDT < 0.43 WP
                    #    jmstau21DDTDownVal = 0.990
                    #    jmstau21DDTUpVal = 1.010
                    #    self.jetSmearer.jmr_vals = [1.124, 1.208, 1.040]
                    #elif "UL2016" in self.era: #KD-using 2016 values from above, not UL2016, since not available yet, as of 29/08/22, and it seems the official recipe does the same as I do for UL17/18 vs. 17/18
                    #    jmstau21DDTNomVal = 1.014
                    #    jmstau21DDTDownVal = 1.007
                    #    jmstau21DDTUpVal = 1.021
                    #    self.jetSmearer.jmr_vals = [1.086, 1.176, 0.996]
                    #elif self.era in ["UL2017"]:
                    #    jmstau21DDTNomVal = 0.983
                    #    jmstau21DDTDownVal = 0.976
                    #    jmstau21DDTUpVal = 0.99
                    #    self.jetSmearer.jmr_vals = [1.080, 1.161, 0.999]
                    #elif self.era in ["UL2018"]:
                    #    jmstau21DDTNomVal = 1.000  # tau21DDT < 0.43 WP
                    #    jmstau21DDTDownVal = 0.990
                    #    jmstau21DDTUpVal = 1.010
                    #    self.jetSmearer.jmr_vals = [1.124, 1.208, 1.040]    
                    #(jet_msdcorr_tau21DDT_jmrNomVal,
                    # jet_msdcorr_tau21DDT_jmrUpVal,
                    # jet_msdcorr_tau21DDT_jmrDownVal
                    # ) = self.jetSmearer.getSmearValsM(
                    #     groomedP4, genGroomedJet
                    # ) if groomedP4 is not None and genGroomedJet is not None else (0., 0., 0.)#

                    #jet_msdcorr_tau21DDT_nom = jet_pt_jerNomVal * \
                    #    jet_msdcorr_tau21DDT_jmrNomVal * jmstau21DDTNomVal * jet_msdcorr_raw
                    #jets_msdcorr_tau21DDT_nom.append(jet_msdcorr_tau21DDT_nom)

                    #jet_msdcorr_tau21DDT_jerUp = {
                    #    jerID: jet_msdcorr_tau21DDT_nom
                    #    for jerID in self.splitJERIDs
                    #}
                    #jet_msdcorr_tau21DDT_jerDown = {
                    #    jerID: jet_msdcorr_tau21DDT_nom
                    #    for jerID in self.splitJERIDs
                    #}
                    #jet_msdcorr_tau21DDT_jerUp[thisJERID] = jet_pt_jerUpVal * \
                    #    jet_msdcorr_tau21DDT_jmrNomVal * jmstau21DDTNomVal * jet_msdcorr_raw
                    #jet_msdcorr_tau21DDT_jerDown[thisJERID] = jet_pt_jerDownVal * \
                    #    jet_msdcorr_tau21DDT_jmrNomVal * jmstau21DDTNomVal * jet_msdcorr_raw
                    #for jerID in self.splitJERIDs:
                    #    jets_msdcorr_tau21DDT_jerUp[jerID].append(
                    #        jet_msdcorr_tau21DDT_jerUp[jerID])
                    #    jets_msdcorr_tau21DDT_jerDown[jerID].append(
                    #        jet_msdcorr_tau21DDT_jerDown[jerID])
                    #jets_msdcorr_tau21DDT_jmrUp.append(
                    #    jet_pt_jerNomVal * jet_msdcorr_tau21DDT_jmrUpVal *
                    #    jmstau21DDTNomVal * jet_msdcorr_raw)
                    #jets_msdcorr_tau21DDT_jmrDown.append(
                    #    jet_pt_jerNomVal * jet_msdcorr_tau21DDT_jmrDownVal *
                    #    jmstau21DDTNomVal * jet_msdcorr_raw)
                    #jets_msdcorr_tau21DDT_jmsUp.append(
                    #    jet_pt_jerNomVal * jet_msdcorr_tau21DDT_jmrNomVal *
                    #    jmstau21DDTUpVal * jet_msdcorr_raw)
                    #jets_msdcorr_tau21DDT_jmsDown.append(
                    #    jet_pt_jerNomVal * jet_msdcorr_tau21DDT_jmrNomVal *
                    #    jmstau21DDTDownVal * jet_msdcorr_raw)

                    # Restore original jmr_vals in jetSmearer
                    self.jetSmearer.jmr_vals = self.jmrVals

            if not(self.isData):
                # evaluate JES uncertainties
                jet_pt_jesUp = {}
                jet_pt_jesDown = {}
                jet_mass_jesUp = {}
                jet_mass_jesDown = {}
                jet_msdcorr_jesUp = {}
                jet_msdcorr_jesDown = {}

                for jesUncertainty in self.jesUncertainties:
                    # (cf. https://twiki.cern.ch/twiki/bin/view/CMSPublic/WorkBookJetEnergyCorrections#JetCorUncertainties)
                    # cf. https://hypernews.cern.ch/HyperNews/CMS/get/JetMET/2000.html
                    if jesUncertainty == "HEMIssue":

                        delta = 1.
                        if jet_pt_nom > 15 and jet.jetId & 2 and jet.phi > -1.57 and jet.phi < -0.87:
                            if jet.eta > -2.5 and jet.eta < -1.3:
                                delta = 0.8
                            elif jet.eta <= -2.5 and jet.eta > -3:
                                delta = 0.65

                        jet_pt_jesUp[jesUncertainty] = jet_pt_nom
                        jet_pt_jesDown[jesUncertainty] = delta * jet_pt_nom
                        jet_mass_jesUp[jesUncertainty] = jet_mass_nom
                        jet_mass_jesDown[jesUncertainty] = delta * jet_mass_nom

                        if self.doGroomed:
                            jet_msdcorr_jesUp[jesUncertainty] = jet_msdcorr_nom
                            jet_msdcorr_jesDown[jesUncertainty] = delta * jet_msdcorr_nom
                    else:

                        self.jesUncertainty[jesUncertainty].setJetPt(jet_pt_nom)
                        self.jesUncertainty[jesUncertainty].setJetEta(jet.eta)

                        delta = self.jesUncertainty[jesUncertainty].getUncertainty(True)

                        jet_pt_jesUp[jesUncertainty] = jet_pt_nom * (1. + delta)

                        jet_pt_jesDown[jesUncertainty] = jet_pt_nom * (1. - delta)

                        jet_mass_jesUp[jesUncertainty] = jet_mass_nom * (1. + delta)

                        jet_mass_jesDown[jesUncertainty] = jet_mass_nom * (1. - delta)

                        if self.doGroomed:
                            jet_msdcorr_jesUp[jesUncertainty] = jet_msdcorr_nom * (1. + delta)

                            jet_msdcorr_jesDown[jesUncertainty] = jet_msdcorr_nom * (1. - delta)

                    jets_pt_jesUp[jesUncertainty].append(jet_pt_jesUp[jesUncertainty])
                    jets_pt_jesDown[jesUncertainty].append(jet_pt_jesDown[jesUncertainty])
                    jets_mass_jesUp[jesUncertainty].append(jet_mass_jesUp[jesUncertainty])
                    jets_mass_jesDown[jesUncertainty].append(jet_mass_jesDown[jesUncertainty])

                    if self.doGroomed:
                        jets_msdcorr_jesUp[jesUncertainty].append(jet_msdcorr_jesUp[jesUncertainty])
                        jets_msdcorr_jesDown[jesUncertainty].append(jet_msdcorr_jesDown[jesUncertainty])

        self.out.fillBranch("%s_pt_raw" % self.jetBranchName, jets_pt_raw)
        self.out.fillBranch("%s_pt_nom" % self.jetBranchName, jets_pt_nom)
        self.out.fillBranch("%s_corr_JEC" % self.jetBranchName, jets_corr_JEC)
        self.out.fillBranch("%s_mass_raw" % self.jetBranchName, jets_mass_raw)
        self.out.fillBranch("%s_mass_nom" % self.jetBranchName, jets_mass_nom)

        if not(self.isData):
            self.out.fillBranch("%s_corr_JER" % self.jetBranchName,
                                jets_corr_JER)
            self.out.fillBranch("%s_corr_JMS" % self.jetBranchName,
                                jets_corr_JMS)
            #print(genAK8_msd_from_two_subjets)

            #self.out.fillBranch("%s_msoftdrop_two_subjets" % self.genJetBranchName,
            #                     genAK8_msd_from_two_subjets )
            #self.out.fillBranch("%s_msoftdrop_three_subjets" % self.genJetBranchName,
            #                     genAK8_msd_from_three_subjets )
            #self.out.fillBranch("%s_msoftdrop_four_subjets" % self.genJetBranchName,
            #                     genAK8_msd_from_four_subjets )
            #self.out.fillBranch("%s_msoftdrop_five_subjets" % self.genJetBranchName,
            #                     genAK8_msd_from_five_subjets )

            if self.applyAK8JMRSmearing:
                self.out.fillBranch("%s_corr_JMR" % self.jetBranchName,
                                    jets_corr_JMR)

            
            for jerID in self.splitJERIDs:
                self.out.fillBranch(
                    "%s_pt_jer%sUp" % (self.jetBranchName, jerID),
                    jets_pt_jerUp[jerID])
                self.out.fillBranch(
                    "%s_pt_jer%sDown" % (self.jetBranchName, jerID),
                    jets_pt_jerDown[jerID])
                self.out.fillBranch(
                    "%s_mass_jer%sUp" % (self.jetBranchName, jerID),
                    jets_mass_jerUp[jerID])
                self.out.fillBranch(
                    "%s_mass_jer%sDown" % (self.jetBranchName, jerID),
                    jets_mass_jerDown[jerID])

            if self.applyAK8JMRSmearing:
                self.out.fillBranch("%s_mass_jmrUp" % self.jetBranchName,
                                    jets_mass_jmrUp)
                self.out.fillBranch("%s_mass_jmrDown" % self.jetBranchName,
                                    jets_mass_jmrDown)
            
            self.out.fillBranch("%s_mass_jmsUp" % self.jetBranchName,
                                jets_mass_jmsUp)
            self.out.fillBranch("%s_mass_jmsDown" % self.jetBranchName,
                                jets_mass_jmsDown)

        if self.doGroomed:
            self.out.fillBranch("%s_msoftdrop_raw" % self.jetBranchName,
                                jets_msdcorr_raw)
            self.out.fillBranch("%s_msoftdrop_nom" % self.jetBranchName,
                                jets_msdcorr_nom)
            self.out.fillBranch("%s_msoftdrop_corr_subjetJEC" % self.jetBranchName,
                                jets_msdcorr_corr_subjetJEC)
            self.out.fillBranch("%s_msoftdrop_nom_PUPPICorred" % self.jetBranchName, 
                            jets_msdcorr_nom_PUPPICorred)

            self.out.fillBranch("%s_msoftdrop_corr_JMS" % self.jetBranchName,
                                jets_msdcorr_corr_JMS)
            
            self.out.fillBranch("%s_msoftdrop_corr_PUPPI" % self.jetBranchName,
                                jets_msdcorr_corr_PUPPI)
            
            if not(self.isData):
                if self.applyAK8JMRSmearing:
                    self.out.fillBranch("%s_msoftdrop_corr_JMR" % self.jetBranchName,
                                        jets_msdcorr_corr_JMR)
                #self.out.fillBranch(
                #    "%s_msoftdrop_tau21DDT_nom" % self.jetBranchName,
                #     jets_msdcorr_tau21DDT_nom)
                for jerID in self.splitJERIDs:
                    self.out.fillBranch(
                        "%s_msoftdrop_jer%sUp" % (self.jetBranchName, jerID),
                        jets_msdcorr_jerUp[jerID])
                    self.out.fillBranch(
                        "%s_msoftdrop_jer%sDown" % (self.jetBranchName, jerID),
                        jets_msdcorr_jerDown[jerID])
                    #self.out.fillBranch(
                    #    "%s_msoftdrop_tau21DDT_jer%sUp" %
                    #    (self.jetBranchName, jerID),
                    #    jets_msdcorr_tau21DDT_jerUp[jerID])
                    #self.out.fillBranch(
                    #    "%s_msoftdrop_tau21DDT_jer%sDown" %
                    #    (self.jetBranchName, jerID),
                    #    jets_msdcorr_tau21DDT_jerDown[jerID])
                if self.applyAK8JMRSmearing:
                    
                    self.out.fillBranch("%s_msoftdrop_jmrUp" % self.jetBranchName,
                                        jets_msdcorr_jmrUp)
                    self.out.fillBranch(
                        "%s_msoftdrop_jmrDown" % self.jetBranchName,
                        jets_msdcorr_jmrDown)

                self.out.fillBranch("%s_msoftdrop_jmsUp" % self.jetBranchName,
                                    jets_msdcorr_jmsUp)
                self.out.fillBranch(
                    "%s_msoftdrop_jmsDown" % self.jetBranchName,
                    jets_msdcorr_jmsDown)

                #self.out.fillBranch(
                #    "%s_msoftdrop_tau21DDT_jmrUp" % self.jetBranchName,
                #    jets_msdcorr_tau21DDT_jmrUp)
                #self.out.fillBranch(
                #    "%s_msoftdrop_tau21DDT_jmrDown" % self.jetBranchName,
                #    jets_msdcorr_tau21DDT_jmrDown)
                #self.out.fillBranch(
                #    "%s_msoftdrop_tau21DDT_jmsUp" % self.jetBranchName,
                #    jets_msdcorr_tau21DDT_jmsUp)
                #self.out.fillBranch(
                #    "%s_msoftdrop_tau21DDT_jmsDown" % self.jetBranchName,
                #    jets_msdcorr_tau21DDT_jmsDown)

        if not(self.isData):
            for jesUncertainty in self.jesUncertainties:
                self.out.fillBranch(
                    "%s_pt_jes%sUp" % (self.jetBranchName, jesUncertainty),
                    jets_pt_jesUp[jesUncertainty])
                self.out.fillBranch(
                    "%s_pt_jes%sDown" % (self.jetBranchName, jesUncertainty),
                    jets_pt_jesDown[jesUncertainty])
                self.out.fillBranch(
                    "%s_mass_jes%sUp" % (self.jetBranchName, jesUncertainty),
                    jets_mass_jesUp[jesUncertainty])
                self.out.fillBranch(
                    "%s_mass_jes%sDown" % (self.jetBranchName, jesUncertainty),
                    jets_mass_jesDown[jesUncertainty])

                if self.doGroomed:
                    self.out.fillBranch(
                        "%s_msoftdrop_jes%sUp" %
                        (self.jetBranchName, jesUncertainty),
                        jets_msdcorr_jesUp[jesUncertainty])
                    self.out.fillBranch(
                        "%s_msoftdrop_jes%sDown" %
                        (self.jetBranchName, jesUncertainty),
                        jets_msdcorr_jesDown[jesUncertainty])

        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed
fatJetUncertainties2016 = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"])
fatJetUncertainties2016All = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"])

fatJetUncertainties2017 = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"])
fatJetUncertainties2017All = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"])

fatJetUncertainties2018 = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"])
fatJetUncertainties2018All = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"])

fatJetUncertainties2016AK4Puppi = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"], jetType="AK4PFPuppi")
fatJetUncertainties2016AK4PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK4PFPuppi")

fatJetUncertainties2017AK4Puppi = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"], jetType="AK4PFPuppi")
fatJetUncertainties2017AK4PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK4PFPuppi")

fatJetUncertainties2018AK4Puppi = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"], jetType="AK4PFPuppi")
fatJetUncertainties2018AK4PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"], jetType="AK4PFPuppi")

fatJetUncertainties2016AK8Puppi = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["Total"], jetType="AK8PFPuppi")
fatJetUncertainties2016AK8PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2016", "Summer16_07Aug2017_V11_MC", ["All"], jetType="AK8PFPuppi")
fatJetUncertainties2016AK8PuppiNoGroom = lambda: fatJetUncertaintiesProducer(
    "2016",
    "Summer16_07Aug2017_V11_MC", ["Total"],
    jetType="AK8PFPuppi",
    noGroom=True)
fatJetUncertainties2016AK8PuppiAllNoGroom = lambda: fatJetUncertaintiesProducer(
    "2016",
    "Summer16_07Aug2017_V11_MC", ["All"],
    jetType="AK8PFPuppi",
    noGroom=True)

fatJetUncertainties2017AK8Puppi = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["Total"], jetType="AK8PFPuppi")
fatJetUncertainties2017AK8PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2017", "Fall17_17Nov2017_V32_MC", ["All"], jetType="AK8PFPuppi")

fatJetUncertainties2018AK8Puppi = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["Total"], jetType="AK8PFPuppi")
fatJetUncertainties2018AK8PuppiAll = lambda: fatJetUncertaintiesProducer(
    "2018", "Autumn18_V8_MC", ["All"], jetType="AK8PFPuppi")
