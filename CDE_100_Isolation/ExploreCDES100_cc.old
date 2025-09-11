// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000*         // If Want more events



// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <string>
#include <vector>
#include <cmath>

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
bool keep_pixel(double theta1_deg, double phi1_deg,
               double theta2_deg, double phi2_deg,
               double threshold_deg = 2.0);

//----------------------------------------------------------------------
using namespace std; // So the compiler interprets cout as std::cout

// ----------------------------------------------------------------------
// Globals

const double kPi = TMath::Pi();
const double nanosecond = 1;
const double meter = 1;
const double kSpeedOfLight = 0.299792458 * meter/nanosecond; // in units if m/ns


// ----------------------------------------------------------------------

int 
main (int argc, char **argv)
{
  //Setup 
  gErrorIgnoreLevel = kError; // Ignore ROOT errors
  
  
  cout << "TODO : Check that EventClass is actually the same for HEAT and HECO" << endl;
  cout << "TODO : Add event Descriptors like Cherenkov Fraction, GenValue is also available, think of other things" << endl;
  cout << "TODO : I am still incorrectly polling the Pixel Pointing Directions. Check with Bruce? maybe he has reference on that?" << endl;
  
  
  
  
  // Begin actual main Execution
  // Check if the user specified what files to analyze
  if (argc < 2) {
    cout << "Usage: ./ReadADST [options] $filename(s)" << endl;
    cout << "Options:" << endl;
    cout << "  --verbose, -v    Enable verbose output" << endl;
    return 1;
  }
  
  // Verbosity flag
  bool superverbose = false; // For debugging
  bool verbose = false;

  // Parse command-line arguments
  vector<string> filenames;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
        verbose = true; // Enable verbose output
    } else if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".root") {
        filenames.push_back(arg); // Treat as a .root filename
    } else if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".txt") {
        // Handle .txt file containing filenames
        ifstream txtFile(arg);
        if (!txtFile.is_open()) {
            cerr << "Error: Could not open file " << arg << endl;
            return 1;
        }
        string line;
        while (getline(txtFile, line)) {
            if (!line.empty()) {
                if (line.size() > 5 && line.substr(line.size() - 5) == ".root") {
                    filenames.push_back(line); // Add valid .root filename
                } else {
                    cerr << "Warning: Skipping invalid line in " << arg << ": " << line << endl;
                }
            }
        }
        txtFile.close();
    } else {
        cerr << "Warning: Unrecognized file type for " << arg << ". Skipping." << endl;
    }
  }

  // Check if no files were provided
  if (filenames.empty()) {
    cout << "Error: No input files provided." << endl;
    return 1;
  }

  if (verbose) {cout << "Verbose mode enabled." << endl;}

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
      if (verbose) {cout << "        Loading File Event " << iEvent;}

      // The length of vector must be exactly 2, Need HEAT only events here.
      if (fdEvents.size() != 2) {
        if (verbose) {cout << " - has " << fdEvents.size() << " FDEvents - Skipping." << endl;}
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
          if (eye->GetMirrorsInEye() != 1) continue ; // only interersted in events with exactly one mirror. and that mirror has to be HEAT mirror
          HC_Event = *eye;                            // That is guaranteed by the above check
          GotHC = true;
          if (superverbose && verbose) {cout << "Got HC Event" << endl;}
          // Check that there is exactly one telescope present int this event
        }
      } // Computation i will do outside of the loop
      if (!GotHE || !GotHC) {
        if (verbose) {cout << " - does not have both HE and HC events. Skipping." << endl;}
        continue;
      }
      // ----------------------------------------------------------------- If Got the data, can process.

      NEvent++;  // Only do that after the Eye values have been read (dont want to add everything to even level data and then have the eye fail)
      if (verbose) {
        cout << endl << "Event " << NEvent << endl;
      } else {
        cout << "Event " << NEvent << "\r" << flush;
      }
      

      
      // Now we go through the pixels. 
      // Writing into file will be done in the loop, so not necessaty to make a new scope.
      // Here we write the data to the file
      FdRecPixel & RecPixel = HC_Event.GetFdRecPixel();
      unsigned int Total_Pixels_inEvent   = RecPixel.GetNumberOfPixels();
      unsigned int Total_Pulsed_Pixels    = RecPixel.GetNumberOfSDPFitPixels(); // Should USe GetNumberOfPulsedPixels(); but can cut some shit events by using this
      if (Total_Pixels_inEvent == 0) {
        if (verbose) {cout << "Event " << NEvent << " has no pixels. Skipping." << endl;}
        continue;
      }

      Pixel_Start = NPixel; // Row of pixel datafile to which the first pixel in event will be written to
      for (unsigned int iPix = 0; iPix < Total_Pixels_inEvent; iPix++) {
        Status = RecPixel.GetStatus(iPix);
        
      }
      // Clean Up 
    } // Event Loop

    if (verbose) {
      cout << "Finished processing file: " << file << endl;
    }
    delete theRecEvent;
    delete detGeometry;
  } // File Loop

  cout << "All files processed successfully, with total of " << NEvent << " events and " << NPixel << " pixels." << endl;
  
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

constexpr double deg2rad(double deg) {
  return deg * M_PI / 180.0;
}

bool keep_pixel(double theta1_deg, double phi1_deg,
                double theta2_deg, double phi2_deg,
                double threshold_deg) {
double theta1 = deg2rad(theta1_deg);
double phi1   = deg2rad(phi1_deg);
double theta2 = deg2rad(theta2_deg);
double phi2   = deg2rad(phi2_deg);
double threshold = deg2rad(threshold_deg);

double cos_ang = std::sin(theta1) * std::sin(theta2) * std::cos(phi1 - phi2) +
       std::cos(theta1) * std::cos(theta2);
if (cos_ang >  1.0 ) cos_ang = 1.0;
if (cos_ang < -1.0 ) cos_ang =-1.0; // Domain check
double angle = std::acos(cos_ang);  // in radians
return angle <= threshold;
}

