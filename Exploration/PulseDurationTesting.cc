// transform Gen geometries (e.g. SDP) from HECO reference to HE reference.
//  BRD 13 March 2025
// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000*         // If Want more events



// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <string>
#include <iomanip> // Include this for std::setw

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
  gROOT->SetStyle("Plain");
  gStyle->SetPadGridX(true); //   This makes all plots drawn have X and Y grid lines.
  gStyle->SetPadGridY(true);

  gStyle->SetLineWidth(1);
  const int canvasX = 800; // pixels
  const int canvasY = 600;

  TH1F h_PulseDuration("h_PulseDuration", "PulseDuration", 50, 0,50);
  TH1F h_TotalEventDuration("h_TotalEventDuration", "EventDuration", 50, 0,100);
  //   TH1F h_SDP_Diff_PCGF  ("h_SDP_Diff_PCGF"  , "SDP SpaceAngle diff (HECO)" , 40,  0,10);
  //   TH1F h_Chi0_Diff_PCGF ("h_Chi0_Diff_PCGF" , "Chi0 diff (HECO)"           , 40, -10,10);
  //   TH1F h_Rp_Diff_PCGF   ("h_Rp_Diff_PCGF"   , "Rp   diff (HECO)"           , 40, -5000,5000);
  //   TH1F h_Axis_Diff_PCGF ("h_Axis_Diff_PCGF" , "Axis SpaceAngle diff (HECO)", 40,  0,10);

  
// Going to write to file instead
// std::ofstream outFile("PCGF_Performance.csv");
// outFile << "SDP_SpaceAngle_Diff,Chi0_Diff,Rp_Diff,Axis_SpaceAngle_Diff,Rec_Level,Shower_Class" << std::endl;

  
  

  // ----------------------------------------------------------------------
  // Loop over files
  // read the file as the second to last argument
  unsigned int NTotalEvents = 0;
  unsigned int NTotalPixels = 0;
  unsigned int NTotalTimeFitPixels =0;

  for (int iFile = 1; iFile <= argc - 1; iFile++) { // File Loop

    RecEventFile dataFile(argv[iFile]);
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++) { // Event Loop
	  //  for (unsigned int iEvent = 0; iEvent < 10000; iEvent++) { // Also Event Loop, but exit at 10000
      if (NTotalEvents % 100 == 0) {
        cout << "Processing event " << NTotalEvents << endl;
      }
      
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // move to next file if no more events, Command Does the reading
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      // The length of vector must be exactly 2, Need HEAT only events here.
      if (fdEvents.size() != 2)continue;
      for (std::vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) { // Eye loop
        
        const unsigned int eyeID=eye->GetEyeId();
        if (eyeID == 5) {
          if (eye->GetMirrorsInEye() != 1) continue ; // only interersted in events with exactly one mirror. and that mirror has to be HEAT mirror
          FdRecPixel & RecPixel = eye->GetFdRecPixel();

          unsigned int MaxPulseStop = 0;
          unsigned int MinPulseStart = 20000;
          unsigned int NPixels = RecPixel.GetNumberOfTracePixels();

          for (unsigned int iPix = 0; iPix < NPixels; iPix++) {
            unsigned int Status = RecPixel.GetStatus(iPix);
            if (Status == 4) { // Only process pixels with a triggered pulse
              NTotalTimeFitPixels++;
              unsigned int PulseStart = RecPixel.GetPulseStart(iPix);
              unsigned int PulseStop  = RecPixel.GetPulseStop(iPix);
              if (PulseStart < MinPulseStart) MinPulseStart = PulseStart;
              if (PulseStop > MaxPulseStop) MaxPulseStop = PulseStop;
              

              // cout << "Pixel " << std::setw(5) << iPix 
              //      << " Status: " << std::setw(5) << Status
              //      << " Start: " << std::setw(5) << PulseStart 
              //      << " Stop: " << std::setw(5) << PulseStop 
              //      << " Duration: " << std::setw(5) << (PulseStop - PulseStart) << endl;
              // // Put Pulse Duration into Histogram
              h_PulseDuration.Fill(PulseStop - PulseStart);
            } else {NTotalPixels++;}
          } // Pixel Loop
          // cout << endl;
          // Put Total Event Duration into Histogram
          unsigned int PulseDuration = MaxPulseStop - MinPulseStart;
          if (PulseDuration > 0) {
            h_TotalEventDuration.Fill(PulseDuration);
          }

          NTotalEvents++;
        } else continue;

      } // Eye Loop
    } // Event Loop
  } // File Loop

  
  // A TApplication is required for drawing
  //move to top of program, and modify
  // TApplication theApp("app", &argc, argv);  
  // Draw the Histograms
  TCanvas c_PulseDuration("c_PulseDuration", " ", canvasX, canvasY);
  h_PulseDuration.Draw();
  gPad->SetLogy();
  TCanvas c_TotalEventDuration("c_TotalEventDuration", " ", canvasX, canvasY);
  h_TotalEventDuration.Draw();
  gPad->SetLogy();
  // Save the Histograms
  c_PulseDuration.SaveAs("PulseDuration.png");
  c_TotalEventDuration.SaveAs("TotalEventDuration.png");
  
  // Print out the number of events and pixels
  cout << "Total Events: " << NTotalEvents << endl;
  cout << "Total Pixels: " << NTotalPixels << endl;
  cout << "Total Time Fit Pixels: " << NTotalTimeFitPixels << endl;
  
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


