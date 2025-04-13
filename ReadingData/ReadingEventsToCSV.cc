// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000*         // If Want more events



// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <string>
#include <vector>

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


// Functions Defined at the bottom - Help with data Parsing, and usage
void usage();
unsigned int ShortPrimaryInt(unsigned int Primary);
unsigned int EventClass_int(string EventClass);


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
  //Setup 
  bool superverbose = true; // For debugging
  gErrorIgnoreLevel = kError; // Ignore ROOT errors


  cout << "TODO : Check that EventClass is actually the same for HEAT and HECO" << endl;
  cout << "TODO : Add event Descriptors like Cherenkov Fraction, GenValue is also available, think of other things" << endl;
  
  
  
  
  // Begin actual main Execution
  // Check if the user specified what files to analyze
  if (argc < 2) {
    cout << "Usage: ./ReadADST [options] $filename(s)" << endl;
    cout << "Options:" << endl;
    cout << "  --verbose, -v    Enable verbose output" << endl;
    return 1;
  }

  // Verbosity flag
  bool verbose = false;

  // Parse command-line arguments
  vector<string> filenames;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
      verbose = true; // Enable verbose output
    } else {
      filenames.push_back(arg); // Treat as a filename
    }
  }

  // Check if no files were provided
  if (filenames.empty()) {
    cout << "Error: No input files provided." << endl;
    return 1;
  }

  if (verbose) {cout << "Verbose mode enabled." << endl;}

  // Initialise the output files
  // Make sure they are empty i.e. overwrite old data
  std::ofstream EventLevelData("EventLevelData.csv");
  std::ofstream PixelLevelData("PixelLevelData.csv");
  
  // Things that will be in the EventLevelData
  EventLevelData << "EventID,Rec_Level,Event_Class,Primary"; // Event Meta Data
  EventLevelData << "Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_dEdXmax,Gen_SDPPhi,Gen_SDPTheta,Gen_Chi0,Gen_Rp,Gen_T0,Gen_CoreEyeDist,"; // Gen Observables
  EventLevelData << "Rec_LogE,Rec_CosZenith,Rec_Xmax,Rec_dEdXmax,Rec_SDPPhi,Rec_SDPTheta,Rec_Chi0,Rec_Rp,Rec_T0,Rec_CoreEyeDist,"; // Rec Observables
  EventLevelData << "Pixel_Start,Pixel_End"; // CSV Reading Information
  // Things that will be in the PixelLevelData
  // Pixel Data, Tracebins will be an array separated by not-commas
  PixelLevelData << "PixelID,TelID,EyeID,Status,Charge,Theta,Phi,TimeOffset,PulseStart,PulseCentroid,PulseEnd,TraceBins"; 
  


  // ------------------------------------------------------
  // Reading data here for real
  // Run the loop over files and events
  unsigned int NEvent = 0;
  unsigned int NPixel = 0; // Num of pixels in the PixelLevelData, use this to specify Pixel_Start and Pixel_End in the EventLevelData file

  for (const string& file : filenames) {
    if (verbose) {
      cout << "Processing file: " << file << endl;
    }

    RecEventFile dataFile(file.c_str());
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    // Check if the rec Event really does exit
    cout << "The Rec Event: " << theRecEvent << endl;

    unsigned int ntotThisFile = dataFile.GetNEvents();
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // Move to next file if no more events
      if (superverbose && verbose ) {cout << "Trying to read DataObjects for event " << NEvent+1 <<  endl;}

      FDEvent HE_Event;
      FDEvent HC_Event;
      bool GotHE = false;
      bool GotHC = false;
      
      // ----------------------------------------------------------------- Getting the FD Event Data
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      // The length of vector must be exactly 2, Need HEAT only events here.
      if (fdEvents.size() != 2) {
        if (verbose) {cout << "Event " << iEvent << " has " << fdEvents.size() << " FDEvents, not 2. Skipping." << endl;}
        continue;
      }
      for (std::vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {
        
        const unsigned int eyeID=eye->GetEyeId();

        if (eyeID == 5) {
          HE_Event = *eye;
          GotHE = true;
          if (superverbose && verbose) {cout << "Got HE Event" << endl;}
        }
        if (eyeID == 6) {
          HC_Event = *eye;
          GotHC = true;
          if (superverbose && verbose) {cout << "Got HC Event" << endl;}
        }
      } // Computation i will do outside of the loop
      if (!GotHE || !GotHC) {
        if (verbose) {cout << "Event " << iEvent << " does not have both HE and HC events. Skipping." << endl;}
        continue;
      }
      // ----------------------------------------------------------------- If Got the data, can process.

      NEvent++;  // Only do that after the Eye values have been read (dont want to add everything to even level data and then have the eye fail)

      // Preallocate the variables for the event data
      string       EventID     = "";
      unsigned int Rec_Level   = 0;
      unsigned int Event_Class = 0;
      unsigned int Primary     = 0;

      double Gen_LogE        = 0;
      double Gen_CosZenith   = 0;
      double Gen_Xmax        = 0;
      double Gen_dEdXmax     = 0;
      double Gen_SDPPhi      = 0;
      double Gen_SDPTheta    = 0;
      double Gen_Chi0        = 0;
      double Gen_Rp          = 0;
      double Gen_T0          = 0;
      double Gen_CoreEyeDist = 0;

      double Rec_LogE        = 0;
      double Rec_CosZenith   = 0;
      double Rec_Xmax        = 0;
      double Rec_dEdXmax     = 0;
      double Rec_SDPPhi      = 0;
      double Rec_SDPTheta    = 0;
      double Rec_Chi0        = 0;
      double Rec_Rp          = 0;
      double Rec_T0          = 0;
      double Rec_CoreEyeDist = 0;

      unsigned int Pixel_Start = 0;
      unsigned int Pixel_End   = 0;


      // Preallocate the variables for the Pixel Data
      // TODO : make this preallocation

      // -------------------------------------------------------------------
      // Here we collect the data
      // The data is read from 3 objects
      // RecEvent is for metadata and EventLevel things
      // HE_Event is for the HEAT data - mostly the input stuff.
      // HC_Event is for the RecData, which needs to be adjusted, but for now its going to be left alone.
      // ------------------------------------------------- Get things from the recEvent

      if (verbose) {cout << "Event " << NEvent << endl;}
      EventID = theRecEvent->GetEventId();
      if (superverbose && verbose) {cout << "EventID: " << EventID << endl;}
      Rec_Level = HC_Event.GetRecLevel();
      Event_Class = EventClass_int(HE_Event.GetEventClass()); 
      Primary = ShortPrimaryInt(HE_Event.GetGenShower().GetPrimary());


      //  Get Gen Data




      // Here we write the data to the file
      




     
    } // Event Loop

    if (verbose) {
      cout << "Finished processing file: " << file << endl;
    }
  } // File Loop

  if (verbose) {
    cout << "All files processed successfully." << endl;
  }

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

unsigned int ShortPrimaryInt(unsigned int Primary) {
    if (Primary == 1000026056) return 26056;
    if (Primary == 1000008016) return 8016;
    if (Primary == 1000002004) return 2004;
    if (Primary == 1000007014) return 7014;
    return Primary;
}

unsigned int EventClass_int(string EventClass) {
    // All possible Event Types (i think)
    // ['Shower Candidate', 'Close Shower', 'Horizontal Shower' , 'Large Event'] 
    // ['Muon + Noise', 'Long Muon', 'Noise', 'Muon']
    if (EventClass == "'Shower Candidate'" ) return 0;
    if (EventClass == "'Close Shower'"     ) return 1;
    if (EventClass == "'Horizontal Shower'") return 2;
    if (EventClass == "'Large Event'"      ) return 3;
    if (EventClass == "'Muon + Noise'"     ) return 4;
    if (EventClass == "'Long Muon'"        ) return 5;
    if (EventClass == "'Noise'"            ) return 6;
    if (EventClass == "'Muon'"             ) return 7;
    return -1; // return -1 if the event class is not recognised
}
