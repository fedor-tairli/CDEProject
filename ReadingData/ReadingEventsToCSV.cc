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

  // Initialise the output files
  // Make sure they are empty i.e. overwrite old data
  std::ofstream EventLevelData("EventLevelData.csv");
  std::ofstream PixelLevelData("PixelLevelData.csv");
  
  // Things that will be in the EventLevelData
  EventLevelData << "EventID,Rec_Level,Event_Class,Primary,"; // Event Meta Data
  EventLevelData << "Gen_LogE,Gen_CosZenith,Gen_Xmax,Gen_dEdXmax,Gen_SDPPhi,Gen_SDPTheta,Gen_Chi0,Gen_Rp,Gen_T0,Gen_CoreEyeDist,Gen_CherenkovFraction,"; // Gen Observables
  EventLevelData << "Rec_LogE,Rec_CosZenith,Rec_Xmax,Rec_dEdXmax,Rec_SDPPhi,Rec_SDPTheta,Rec_Chi0,Rec_Rp,Rec_T0,Rec_CoreEyeDist,Rec_CherenkovFraction,"; // Rec Observables
  EventLevelData << "Pixel_Start,Pixel_End"; // CSV Reading Information
  EventLevelData << endl; // End of header line
  // Things that will be in the PixelLevelData
  // Pixel Data, Tracebins will be an array separated by not-commas
  PixelLevelData << "PixelID,TelID,EyeID,Status,Charge,Theta,Phi,TimeOffset,PulseStart,PulseCentroid,PulseStop,TraceBins"; 
  PixelLevelData << endl; // End of header line
  


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
      // Preallocate the variables for the event data
      // Pretty sure the default values will be returned event if there is nothing, so no need to reset these guys every loop
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
      double Gen_CherenkovFraction = 0;

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
      double Rec_CherenkovFraction = 0;

      unsigned int Pixel_Start = 0;
      unsigned int Pixel_End   = 0;


      // Preallocate the variables for the Pixel Data
      // TODO : make this preallocation
      unsigned int PixelID = 0;
      unsigned int TelID   = 0;
      unsigned int EyeID   = 0;
      unsigned int Status  = 0;
      double Charge = 0;
      double Theta  = 0;
      double Phi    = 0;
      double TimeOffset = 0;
      double PulseStart = 0;
      double PulseCentroid = 0;
      double PulseStop  = 0;
      std::vector<double> TraceBins(2000,0.0); // This is a vector of doubles for GetTrace function.

      // -------------------------------------------------------------------
      // Here we collect the data
      // The data is read from 3 objects
      // RecEvent is for metadata and EventLevel things
      // HE_Event is for the HEAT data - mostly the input stuff.
      // HC_Event is for the RecData, which needs to be adjusted, but for now its going to be left alone.
      // ------------------------------------------------- Get things from the recEvent

      
      EventID = theRecEvent->GetEventId();
      if (superverbose && verbose) {cout << "EventID: " << EventID << endl;}
      Rec_Level = HC_Event.GetRecLevel();
      Event_Class = EventClass_int(HE_Event.GetEventClass()); 
      Primary = ShortPrimaryInt(theRecEvent->GetGenShower().GetPrimary());


      { // Scope for the non Eye variables
        // Event Objects
        GenShower & genShower = theRecEvent->GetGenShower();
        // Event Variables
        Gen_LogE      = genShower.GetEnergy();
        Gen_LogE      = TMath::Log10(Gen_LogE);
        Gen_CosZenith = genShower.GetCosZenith();
        Gen_Xmax      = genShower.GetXmaxInterpolated();
        Gen_dEdXmax   = genShower.GetdEdXmaxInterpolated();
      }        
      { // Scope to get the HEAT Variables
        FdGenGeometry & HE_GenGeometry = HE_Event.GetGenGeometry();
        Gen_SDPPhi      = HE_GenGeometry.GetSDPPhi();
        Gen_SDPTheta    = HE_GenGeometry.GetSDPTheta();
        Gen_Chi0        = HE_GenGeometry.GetChi0();
        Gen_Rp          = HE_GenGeometry.GetRp();
        Gen_T0          = HE_GenGeometry.GetT0();
        Gen_CoreEyeDist = HE_GenGeometry.GetCoreEyeDistance();

        unsigned int MirrorID;
        for (MirrorID = 0; MirrorID <10; MirrorID++){
          if (HE_Event.MirrorIsInEvent(MirrorID)){
            if (verbose && superverbose ) {cout << "Found MirrorID: " << MirrorID << endl;}
            break; // There will only ever be one mirror in these events
          }
        } 
        if (MirrorID == 10 && superverbose && verbose ) {
          cout << "No MirrorID found in this event. Skipping." << endl;
          continue;
        }
        FdTelescopeData & HE_TelData = HE_Event.GetTelescopeData(MirrorID);
        Gen_CherenkovFraction = HE_TelData.GetGenApertureLight().GetCherenkovFraction(); // TODO: NEEDS FIXING something is whack with this thing
        // cout << "                                               Cherenkov Fraction from GenShower : " << Gen_CherenkovFraction << endl;

      }

      { // Scope for HECO (Rec) variables,
        // Note : The Actual values for geometry dont matter (I think) so what we are going to do is:
        //  Calculate :  Gen_HE_value - (Gen_HECO-Rec_HECO) values and that would make a homogenous with the HE Values. 
        
        FdGenGeometry & HECO_GenGeometry = HC_Event.GetGenGeometry();
        FdRecGeometry & HECO_RecGeometry = HC_Event.GetFdRecGeometry();
        // Values that requrie Geometry transformations
        Rec_SDPPhi      = Gen_SDPPhi      - (HECO_GenGeometry.GetSDPPhi()          - HECO_RecGeometry.GetSDPPhi()  );
        Rec_SDPTheta    = Gen_SDPTheta    - (HECO_GenGeometry.GetSDPTheta()        - HECO_RecGeometry.GetSDPTheta());
        Rec_Chi0        = Gen_Chi0        - (HECO_GenGeometry.GetChi0()            - HECO_RecGeometry.GetChi0()    );
        Rec_Rp          = Gen_Rp          - (HECO_GenGeometry.GetRp()              - HECO_RecGeometry.GetRp()      );
        Rec_T0          = Gen_T0          - (HECO_GenGeometry.GetT0()              - HECO_RecGeometry.GetT0()      );
        Rec_CoreEyeDist = Gen_CoreEyeDist - (HECO_GenGeometry.GetCoreEyeDistance() - HECO_RecGeometry.GetCoreEyeDistance());
        // Values that are not affected by telescope geometry
        

        FdRecShower & HECO_RecShower = HC_Event.GetFdRecShower();
        FdApertureLight & HECO_ApertureLight = HC_Event.GetFdRecApertureLight();
        Rec_CherenkovFraction = HECO_ApertureLight.GetCherenkovFraction();

        Rec_LogE              = TMath::Log10(HECO_RecShower.GetEnergy());
        Rec_Xmax              = HECO_RecShower.GetXmax();
        Rec_dEdXmax           = HECO_RecShower.GetdEdXmax();
        // Rec_CosZenith i cannot be bothered to do this, and its probably never going to come up anyway.

      }

      
      // Now we go through the pixels. 
      // Writing into file will be done in the loop, so not necessaty to make a new scope.
      // Here we write the data to the file
      FdRecPixel & RecPixel = HE_Event.GetFdRecPixel();
      unsigned int Total_Pixels_inEvent   = RecPixel.GetNumberOfPixels();
      unsigned int Total_Pulsed_Pixels    = RecPixel.GetNumberOfSDPFitPixels(); // Should USe GetNumberOfPulsedPixels(); but can cut some shit events by using this
      if (Total_Pixels_inEvent == 0) {
        if (verbose) {cout << "Event " << NEvent << " has no pixels. Skipping." << endl;}
        continue;
      }
      Pixel_Start = NPixel; // Row of pixel datafile to which the first pixel in event will be written to
      for (unsigned int iPix = 0; iPix < Total_Pixels_inEvent; iPix++) {
        Status = RecPixel.GetStatus(iPix);
        //Now we hav two cases here - Pulse and No Pulse
        if (Status > 1 ) { // Pulse was already detected so we dont need to do any tricks
          //Pixel Metainfo
          PixelID = RecPixel.GetID(iPix);
          TelID   = RecPixel.GetTelescopeId(iPix);
          EyeID   = 5; // Explicitly is 5, for this analysis
          // Status duh
          // Geometry Info
          EyeGeometry       & HEGeometry  = detGeometry -> GetEye(EyeID);
          TelescopeGeometry & TelGeometry = HEGeometry.GetTelescope(TelID);
          Phi   = TelGeometry.GetPixelPhi(PixelID-440*(TelID-1),"upward");
          Theta = TelGeometry.GetPixelOmega(PixelID-440*(TelID-1),"upward");
          
          // Signal Info
          PulseStart    = RecPixel.GetPulseStart(iPix);
          PulseStop     = RecPixel.GetPulseStop(iPix);
          PulseCentroid = RecPixel.GetTime(iPix);
          TimeOffset    = 0; // -------------------------------------------------------------------- Not actually what it should be, Check With Reference on how to get this
          Charge        = RecPixel.GetCharge(iPix);                                                // Can get away with 0, cause no crossing of mirrors so its always the same. Pretty sure is actually 0
          
          // Collect the Trace
          TraceBins = RecPixel.GetTrace(iPix);
          
        
        } else { // No Pulse, So we need to do some shenanegans. 
          //Pixel Metainfo
          PixelID = RecPixel.GetID(iPix);
          TelID   = RecPixel.GetTelescopeId(iPix);
          EyeID   = 5; // Explicitly is 5, for this analysis
          // Status duh
          // Geometry Info
          EyeGeometry       & HEGeometry  = detGeometry -> GetEye(EyeID);
          TelescopeGeometry & TelGeometry = HEGeometry.GetTelescope(TelID);
          Phi   = TelGeometry.GetPixelPhi(PixelID-440*(TelID-1),"upward");
          Theta = TelGeometry.GetPixelOmega(PixelID-440*(TelID-1),"upward");
          TimeOffset  = 0; // -------------------------------------------------------------------- Not actually what it should be, Check With Reference on how to get this

          // Signal Info -> This is where ths gets interesting
          // ----> We gotta decide what is the min and max time where a pulse exists on the neighbouring pixels.
          // ----> Basically loop over all other pixels, and if they fall into the first category, iteratively make window of Trace Collection larger
          // ----> If there is no pixel close enough, than we discard this pixel and go onto the next one i the main pixel loop.
          // ----> The above check for SDP Fit pixels instead of just pulsed pixels allows us to not care about the noize triggered pulses.
          
          ////////// Pulse Finding 
          PulseStart = 200000;
          PulseStop  = 0;

          for (unsigned int jPix = 0; jPix < Total_Pixels_inEvent; jPix++) {
            unsigned int jPix_Status = RecPixel.GetStatus(jPix);
            if (jPix_Status > 1) { // Found a Pulse pixel
              unsigned int jPix_PixelID = RecPixel.GetID(jPix);
              double jPix_Theta = TelGeometry.GetPixelOmega(jPix_PixelID-440*(TelID-1),"upward"); // Dont have to recall new Geometry objects
              double jPix_Phi   = TelGeometry.GetPixelPhi(jPix_PixelID-440*(TelID-1),"upward");  // Guaranteed to be in the same telescope for this analysis
              // Check if the pixel is close enough to be considered a neighbour
              if (keep_pixel(Theta, Phi, jPix_Theta, jPix_Phi, 2.3)) { // 2 degrees
                double jPix_PulseStart = RecPixel.GetPulseStart(jPix);
                double jPix_PulseStop  = RecPixel.GetPulseStop(jPix);
                if (jPix_PulseStart < PulseStart) {
                  PulseStart = jPix_PulseStart;
                }
                if (jPix_PulseStop > PulseStop) {
                  PulseStop = jPix_PulseStop;
                }
              } else {
                // Do nothing, not a neighbour
              }
            }
          } // jPix Loop
          // Found Pulse I Suppose
          
        
          // Collect the Trace
          TraceBins = RecPixel.GetTrace(iPix);  
          // Now we can check if the pulse start and pulse stop are valid        
          //
          // Lets just say i want to make sure that it is < 24 Bins, (such that i can keep the total of <=30 bins, wich 3 bins each side of the pulse for safety)
          int PulseDuration = PulseStop - PulseStart;
          if (PulseStart < 3 )  {continue;} // That would cause a segfault 
          if (PulseStop > 2000) {continue;} // That would cause a segfault
          // if (PulseDuration <= 0 || PulseDuration >30) {continue;} // Not good pixel

          if (PulseStop - PulseStart < 0  ) {continue;} //without printing, there is too many
          if (PulseStop - PulseStart > 30 ) {
            // cout << "Pulse Duration is too long: " << PulseDuration << endl;
            continue;
          } else {
            // cout << "Pulse is good : " << PulseStop - PulseStart << endl;
          }


          // Gotta construct the Charge, which is basically just going to be a sum of all the bins in TraceBins
          // All meaning the bins in between PulseStart and PulseStop
          Charge = 0;
          for (unsigned int iTraceBin = PulseStart-3; iTraceBin < PulseStop+3; iTraceBin++) {
            Charge += TraceBins[iTraceBin];
          }
          if (Charge < 0) {Charge = 0;}// Sometimes might happen if the trace is noisy, compared to the Total signal in pulse
          PulseCentroid = 0; /// TODO:Check if I can actuallly do this. But for now 0, casue i dont think ill be using it really
        } // Non Triggered Pixel Calculations

        // Now i should have information for both cases of pixels
        // Write the Pixel row here: 
        PixelLevelData << PixelID        << ","; 
        PixelLevelData << TelID          << ","; 
        PixelLevelData << EyeID          << ","; 
        PixelLevelData << Status         << ","; 
        PixelLevelData << Charge         << ",";
        PixelLevelData << Theta          << ",";
        PixelLevelData << Phi            << ",";
        PixelLevelData << TimeOffset     << ",";
        PixelLevelData << PulseStart     << ",";
        PixelLevelData << PulseCentroid  << ",";
        PixelLevelData << PulseStop      << ",";
        
        for (unsigned int iTraceBin = PulseStart-3; iTraceBin < PulseStop+3; iTraceBin++) {
          PixelLevelData << TraceBins[iTraceBin];
          if (iTraceBin < PulseStop+2) { // Space Separation instead of commas
            PixelLevelData << " ";
          } 
        } // Trace Storage Loop
        PixelLevelData << endl; // End of pixel row 
        NPixel++; // Go to next row when pixel was written.
      } // Pixel Loop
      Pixel_End = NPixel; // Row of pixel datafile to which the last pixel in event will be written to (Non-Inclusive)
      // Now we can write the event data to the file
      
      if (Pixel_Start == Pixel_End) {
        if (verbose) {cout << "Event " << NEvent << " has no pixels. Skipping." << endl;}
        continue;
      }
      EventLevelData << EventID        << ","; //
      EventLevelData << Rec_Level      << ","; //// Metadata
      EventLevelData << Event_Class    << ","; //
      EventLevelData << Primary        << ","; //

      EventLevelData << Gen_LogE       << ","; //
      EventLevelData << Gen_CosZenith  << ","; //// Gen Data
      EventLevelData << Gen_Xmax       << ","; //
      EventLevelData << Gen_dEdXmax    << ","; //
      EventLevelData << Gen_SDPPhi     << ","; //
      EventLevelData << Gen_SDPTheta   << ","; //
      EventLevelData << Gen_Chi0       << ","; //
      EventLevelData << Gen_Rp         << ","; //
      EventLevelData << Gen_T0         << ","; //
      EventLevelData << Gen_CoreEyeDist<< ","; //
      EventLevelData << Gen_CherenkovFraction << ",";
      
      EventLevelData << Rec_LogE       << ","; //
      EventLevelData << Rec_CosZenith  << ","; //// Rec Data
      EventLevelData << Rec_Xmax       << ","; //
      EventLevelData << Rec_dEdXmax    << ","; //
      EventLevelData << Rec_SDPPhi     << ","; //
      EventLevelData << Rec_SDPTheta   << ","; //
      EventLevelData << Rec_Chi0       << ","; //
      EventLevelData << Rec_Rp         << ","; //
      EventLevelData << Rec_T0         << ","; //
      EventLevelData << Rec_CoreEyeDist<< ","; //
      EventLevelData << Rec_CherenkovFraction << ",";

      EventLevelData << Pixel_Start    << ",";  // Pixel Positions
      EventLevelData << Pixel_End      << endl; //

      if (verbose) {cout << "      " << NEvent << " written to file." << endl;}

    } // Event Loop

    if (verbose) {
      cout << "Finished processing file: " << file << endl;
    }
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

