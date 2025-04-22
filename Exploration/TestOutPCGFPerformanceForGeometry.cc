// transform Gen geometries (e.g. SDP) from HECO reference to HE reference.
//  BRD 13 March 2025
// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000*         // If Want more events



// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <string>

// Auger
#include "RecEvent.h"
#include "RecEventFile.h"
#include "DetectorGeometry.h"
#include "FileInfo.h"
#include "Detector.h"
#include "SdBadStation.h"
#include "EyeGeometry.h"

// ROOT
#include "TH1F.h"   // Histogram
#include "TH2F.h"   // Histogram
#include "TMath.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TROOT.h"
#include "TStyle.h"
#include "TGraph.h" // Basic scatter plot
#include "TProfile.h"
#include "TLegend.h"

using namespace std;

void usage();

//----------------------------------------------------------------------
using namespace std; // So the compiler interprets cout as std::cout

// ----------------------------------------------------------------------
// Globals

const double kPi = TMath::Pi();
const double nanosecond = 1;
const double meter = 1;
const double kSpeedOfLight = 0.299792458 * meter/nanosecond; // in units if m/ns


int 
main(int argc, char **argv)
{

  // Check if the user specified what files to analyse
  if (argc < 2) {
    usage();
    return 1;
  }

  //Modify call (thanks Violet) to avoid seg fault in newer Offline
  //  TApplication theApp("app", &argc, argv);  
  TApplication theApp("app", NULL, NULL);  


  // Makes your plots look "normal". The default background isn't plain white.
//   gROOT->SetStyle("Plain");
//   gStyle->SetPadGridX(true); //   This makes all plots drawn have X and Y grid lines.
//   gStyle->SetPadGridY(true);

//   gStyle->SetLineWidth(1);
//   const int canvasX = 800; // pixels
//   const int canvasY = 600;

  //  TH1F h_DiffRp("h_DiffRp", "Gen Rp diff (HE-CO)", 40, -500,500);
  //   TH1F h_DiffSDPThetaEye6("h_DiffSDPThetaEye6", "SDPTheta diff (HECO)", 40, -5,5);
//   TH1F h_SDP_Diff_PCGF  ("h_SDP_Diff_PCGF"  , "SDP SpaceAngle diff (HECO)" , 40,  0,10);
//   TH1F h_Chi0_Diff_PCGF ("h_Chi0_Diff_PCGF" , "Chi0 diff (HECO)"           , 40, -10,10);
//   TH1F h_Rp_Diff_PCGF   ("h_Rp_Diff_PCGF"   , "Rp   diff (HECO)"           , 40, -5000,5000);
//   TH1F h_Axis_Diff_PCGF ("h_Axis_Diff_PCGF" , "Axis SpaceAngle diff (HECO)", 40,  0,10);

  
// Going to write to file instead
std::ofstream outFile("PCGF_Performance.csv");
outFile << "SDP_SpaceAngle_Diff,Chi0_Diff,Rp_Diff,Axis_SpaceAngle_Diff,Rec_Level,Shower_Class" << std::endl;

  
  int nQualityEvents = 0; // just a counter

  // ----------------------------------------------------------------------
  // Loop over files
  // read the file as the second to last argument
  unsigned int NTotalEvents = 0;
  for (int iFile = 1; iFile <= argc - 1; iFile++) { // File Loop

    RecEventFile dataFile(argv[iFile]);
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++) { // Event Loop
	  //  for (unsigned int iEvent = 0; iEvent < 10000; iEvent++) { // Also Event Loop, but exit at 10000

      
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // move to next file if no more events, Command Does the reading
      NTotalEvents++;
      if(NTotalEvents%10 ==0) cout << "Read " << NTotalEvents << " total events." << endl;

      //   const string auger_id = theRecEvent->GetEventId();
      // some shower gen parameters
      // const double genEcal = theRecEvent->GetGenShower().GetCalorimetricEnergy();
      const double genE   = theRecEvent->GetGenShower().GetEnergy();
      const int primary   = theRecEvent->GetGenShower().GetPrimary();
      TVector3 CoreSiteCS = theRecEvent->GetGenShower().GetCoreSiteCS();
      TVector3 AxisCoreCS = theRecEvent->GetGenShower().GetAxisCoreCS();
      TVector3 AxisSiteCS = theRecEvent->GetGenShower().GetAxisSiteCS();
      
      
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      for (vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {  // Loop over eyes

        const unsigned int eyeID=eye->GetEyeId();
        if (eyeID != 6) continue; // Only process HECO, thats where the PCGF is 
        FdGenGeometry & GenGeometry = eye->GetGenGeometry();
        FdRecGeometry & RecGeometry = eye->GetFdRecGeometry();
        FdGenShower   & GenShower   = eye->GetGenShower();
        FdRecShower   & RecShower   = eye->GetFdRecShower();

        // SDP - Get values and compute the space angle between them
        TVector3 Rec_SDP = RecGeometry.GetSDP();
        TVector3 Gen_SDP = GenGeometry.GetSDP();
        Rec_SDP = Rec_SDP.Unit();
        Gen_SDP = Gen_SDP.Unit();
        double sdp_space_angle = Rec_SDP.Angle(Gen_SDP)*180./kPi;

        // Chi0 - Get values and compute the difference
        double Rec_Chi0 = RecGeometry.GetChi0()*180./kPi;
        double Gen_Chi0 = GenGeometry.GetChi0()*180./kPi;
        double chi0_diff = Rec_Chi0 - Gen_Chi0;

        // Rp - Get values and compute the difference
        double Rec_Rp = RecGeometry.GetRp();
        double Gen_Rp = GenGeometry.GetRp();
        double Rp_diff = Rec_Rp - Gen_Rp;

        // Axis - Get values and compute the space angle between them
        TVector3 Rec_Axis = RecShower.GetAxisSiteCS();
        TVector3 Gen_Axis = AxisSiteCS;
        Rec_Axis = Rec_Axis.Unit();
        Gen_Axis = Gen_Axis.Unit();
        double axis_space_angle = Rec_Axis.Angle(Gen_Axis)*180./kPi;

        // Get the rest of the data
        int     Rec_Level  = eye->GetRecLevel();
        string Event_Class = eye->GetEventClass();

        // Write data to file instead of histograms
        outFile << sdp_space_angle   << ", " 
                << chi0_diff         << ", " 
                << Rp_diff           << ", " 
                << axis_space_angle  << ", " 
                << Rec_Level         << ", "
                << Event_Class
                << std::endl;

        // // Fill the histograms with the differences
        // h_SDP_Diff_PCGF .Fill(sdp_space_angle);
        // h_Chi0_Diff_PCGF.Fill(chi0_diff);
        // h_Rp_Diff_PCGF  .Fill(Rp_diff);
        // h_Axis_Diff_PCGF.Fill(axis_space_angle);

        } // Eye Loop
      } // Event Loop
    } // File Loop
  outFile.close();

  
//   // A TApplication is required for drawing
//   //move to top of program, and modify
//   //TApplication theApp("app", &argc, argv);  

//   //  TCanvas c_DiffRp("c_DiffRp", " ", canvasX, canvasY);
//   //  h_DiffRp.Draw();
//   TCanvas c_SDP_Diff_PCGF("c_SDP_Diff_PCGF", " ", canvasX, canvasY);
//   gPad->SetLogy(); 
//   h_SDP_Diff_PCGF.Draw();
//   TCanvas c_Chi0_Diff_PCGF("c_Chi0_Diff_PCGF", " ", canvasX, canvasY);
//   gPad->SetLogy();
//   h_Chi0_Diff_PCGF.Draw();
//   TCanvas c_Rp_Diff_PCGF("c_Rp_Diff_PCGF", " ", canvasX, canvasY);
//   gPad->SetLogy();
//   h_Rp_Diff_PCGF.Draw();
//   TCanvas c_Axis_Diff_PCGF("c_Axis_Diff_PCGF", " ", canvasX, canvasY);
//   gPad->SetLogy();
//   h_Axis_Diff_PCGF.Draw();

//   c_SDP_Diff_PCGF  .SaveAs("SDP_Diff_PCGF.png" );
//   c_Chi0_Diff_PCGF .SaveAs("Chi0_Diff_PCGF.png");
//   c_Rp_Diff_PCGF   .SaveAs("Rp_Diff_PCGF.png"  );
//   c_Axis_Diff_PCGF .SaveAs("Axis_Diff_PCGF.png");

} 
  
 

// ----------------------------------------------------------------------
void
usage()
{
  printf("Usage:\n"
  "./ReadADST $filename(s)\n"
  "where $filename is the ADST file(s) you want to analyse.\n"
  "For example, to run over the data from 2011:\n"
  "./ReadADST /remote/kennya/auger/data/ADST/Offline_v2r7p8-Shannon/HybridRec/HybridRec_v7r6_2011*\n"
  "Or just for the month of May, 2011:\n"
  "./ReadADST /remote/kennya/auger/data/ADST/Offline_v2r7p8-Shannon/HybridRec/HybridRec_v7r6_2011_05_generated_2012-10-28.root\n");
}


