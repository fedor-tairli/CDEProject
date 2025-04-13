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


// ----------------------------------------------------------------------

int main (int argc, char **argv)
{
  // Check if the user specified what files to analyse
  if (argc < 2) {
    usage();
    return 1;
  }
  // Some setup for ADST Reading
  TApplication theApp("app", NULL, NULL);

  // Initialise the output files
  // Make sure they are empty i.e. overwrite old data
  std::ofstream EventLevelData('EventLevelData.csv')
  std::ofstream PixelLevelData('PixelLevelData.csv')
  
  // Things that will be in the EventLevelData
  EventLevelData << 'EventID_1,EventID_2,Rec_Level,Event_Class,Primary' // Event Meta Data
  EventLevelData << 'Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_dEdXmax,Gen_SDPPhi,Gen_SDPTheta,Gen_Chi0,Gen_Rp,Gen_T0,Gen_CoreEyeDist,' // Gen Observables
  EventLevelData << 'Rec_LogE,Rec_CosZenith,Rec_Xmax,Rec_dEdXmax,Rec_SDPPhi,Rec_SDPTheta,Rec_Chi0,Rec_Rp,Rec_T0,Rec_CoreEyeDist,' // Rec Observables
  EventLevelData << 'Pixel_Start,Pixel_End' // CSV Reading Information
  // Things that will be in the PixelLevelData
  // Pixel Data, Tracebins will be an array separated by not-commas
  PixelLevelData << 'PixelID,TelID,EyeID,Status,Charge,Theta,Phi,TimeOffset,PulseStart,PulseCentroid,PulseEnd,TraceBins' 

  

  // Run the loop over files and events
  unsigned int NEvent = 0;
  unsigned int NPixel = 0; // This will be the number of pixels in the PixelLevelData, can use this to specify Pixel_Start and Pixel_End in the EventLevelData file
  for (int iFile = 1; iFile <= argc-1; iFile++) {
    RecEventFile dataFile(argv[iFile]);
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // Move to next file if no more events
      NEvent++;







     
    } // Event Loop
  } // File Loop

  return 0;
} // Main Exit




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

