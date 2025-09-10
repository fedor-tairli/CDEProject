// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000*         // If Want more events



// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <sys/stat.h> // For checking if directory exists
#include <sys/types.h> // For mkdir
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

// using namespace std;


// Functions Defined at the bottom - Help with data Parsing, and usage
void usage();
//----------------------------------------------------------------------
// using namespace std; // So the compiler interprets cout as std::cout

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
  
  
  // Begin actual main Execution
  // Check if the user specified what files to analyze
  if (argc < 2) {
    std::cout << "Usage: ./ReadADST [options] $filename(s)" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --verbose, -v    Enable verbose output" << std::endl;
    return 1;
  }
  
  // Verbosity flag
  bool superverbose = true; // For debugging
  bool verbose = false;
  std::string save_file_path = "./ReadEvents/"; // Set up the default save path as relative path

  // Parse command-line arguments
  std::vector<std::string> filenames;
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--verbose" || arg == "-v") {
        verbose = true; // Enable verbose output
    } else if (arg == "--savepath" || arg == "-s") {
        if (i + 1 < argc) {
            save_file_path = argv[++i]; // Get the next argument as the save path
            if (save_file_path.back() != '/') {
                if (save_file_path.size() > 5 && save_file_path.substr(save_file_path.size() - 5) == ".root") {
                    std::cerr << "Error: --savepath option requires a directory path, not a file." << std::endl;
                    return 1;
                } else if (save_file_path.size() > 4 && save_file_path.substr(save_file_path.size() - 4) == ".txt") {
                    std::cerr << "Error: --savepath option requires a directory path, not a file." << std::endl;
                    return 1;
                }
                save_file_path += '/'; // Ensure the path ends with a slash
            }
        } else {
            std::cerr << "Error: --savepath option requires a path argument." << std::endl;
            return 1;
        }

    } else if (arg.size() > 5 && arg.substr(arg.size() - 5) == ".root") {
        filenames.push_back(arg); // Treat as a .root filename
    } else if (arg.size() > 4 && arg.substr(arg.size() - 4) == ".txt") {
        // Handle .txt file containing filenames
        ifstream txtFile(arg);
        if (!txtFile.is_open()) {
            std::cerr << "Error: Could not open file " << arg << std::endl;
            return 1;
        }
        std::string line;
        while (getline(txtFile, line)) {
            if (!line.empty()) {
                if (line.size() > 5 && line.substr(line.size() - 5) == ".root") {
                    filenames.push_back(line); // Add valid .root filename
                } else {
                    std::cerr << "Warning: Skipping invalid line in " << arg << ": " << line << std::endl;
                }
            }
        }
        txtFile.close();
    } else {
        std::cerr << "Warning: Unrecognized file type for " << arg << ". Skipping." << std::endl;
    }
  }

  // Check if no files were provided
  if (filenames.empty()) {
    std::cout << "Error: No input files provided." << std::endl;
    return 1;
  }

  if (verbose) {std::cout << "Verbose mode enabled." << std::endl;}

  // Check if the save path already exist, if not, try to create it
  struct stat info;
  if (stat(save_file_path.c_str(), &info) != 0) {
      if (verbose) {std::cout << "Save path " << save_file_path << " does not exist. Attempting to create it." << std::endl;}
      if (mkdir(save_file_path.c_str(), 0755) != 0) {
          std::cerr << "Error: Could not create save path " << save_file_path << std::endl;
          return 1;
      }
  } else if (!(info.st_mode & S_IFDIR)) {
      std::cerr << "Error: Save path " << save_file_path << " is not a directory." << std::endl;
      return 1;
  }
  
  
  // ------------------------------------------------------
  // Reading Data and re-triggering (will be done in python later)
  // Run the loop over files and events
  unsigned int NEvent_Scanned  = 0;
  unsigned int NEyes_Scanned   = 0;
  unsigned int NEvent_Accepted = 0;

  for (const std::string& file : filenames) {
    if (verbose) {
      std::cout << "Processing file: " << file << std::endl;
    }

    RecEventFile dataFile(file.c_str());
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * det_Geometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*det_Geometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // Move to next file if no more events
      if (superverbose && verbose ) {std::cout << "Trying to read DataObjects for event " << NEvent_Scanned+1 <<  std::endl;}

      
      // ----------------------------------------------------------------- Getting the FD Event Data
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      if (verbose) {std::cout << "        Loading File Event " << iEvent << std::endl;}
      
      NEvent_Scanned += 1;
      for (std::vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {
        const unsigned int eyeID=eye->GetEyeId();
        if (eyeID == 6) continue; // Skip HeCo events for now, Going to deal with variable binning later
        NEyes_Scanned += 1;
        

        
        FdRecPixel & RecPixel = eye->GetFdRecPixel();
        unsigned int Total_Pixels_inEvent   = RecPixel.GetNumberOfPixels();
        unsigned int Total_Pulsed_Pixels    = RecPixel.GetNumberOfSDPFitPixels(); // Should USe GetNumberOfPulsedPixels(); but can cut some shit events by using this
        if (Total_Pulsed_Pixels == 0 ) continue;
        
        // allocate the variables needed for storing event data
        std::vector<std::vector<double>> Signal_array;
        std::vector<std::pair<double, double>> Pos_array;
        
        std::vector<std::vector<bool>> Rec_Trigger_array;
        std::vector<std::vector<bool>> Myy_Trigger_array;
        std::vector<bool> pix_Trigger_array;
        
        std::vector<int> ID_array;
        std::vector<int> Status_array;
        
        // allocate the variables for pixel processing
        std::vector<double> pix_Trace;
        double pix_Theta;
        double pix_Phi;
        int pix_ID;
        int pix_Status;
        
        
        int pix_Geom_ID;
        int pix_tel_ID;
        std::string tel_Pointing_ID;
        double pix_Time_Offset;
        int pix_Bin_Offset;
        double pix_Bin_Width;
        int pix_Pulse_Start;
        int pix_Pulse_Stop;
        int pix_trace_length;
        
        
        
        
        // Now we go through the pixels. 
        for (unsigned int iPix = 0; iPix < Total_Pixels_inEvent; iPix++) {
          
          // Pixel MetaData
          pix_Status      = RecPixel.GetStatus     (iPix);
          pix_ID          = RecPixel.GetID         (iPix);
          pix_Geom_ID     = RecPixel.GetPixelId    (iPix);
          pix_tel_ID      = RecPixel.GetTelescopeId(iPix);
          tel_Pointing_ID = theRecEvent->GetDetector().GetPointingId(eyeID,pix_tel_ID);


          if (!(RecPixel.HasADCTrace(iPix))){
            // Total_Pixels_inEvent -=1;
            continue;
          }

          // Eye and Telescope Geometry
          EyeGeometry       & eye_Geometry  = det_Geometry -> GetEye(eyeID);
          TelescopeGeometry & tel_Geometry = eye_Geometry.GetTelescope(pix_tel_ID);

          pix_Trace = RecPixel.GetTrace(iPix);
          pix_Theta = tel_Geometry.GetPixelOmega(iPix,tel_Pointing_ID);
          pix_Phi   = tel_Geometry.GetPixelPhi  (iPix,tel_Pointing_ID);
          
          pix_trace_length = pix_Trace.size();

          //  Apply the Bin offset to the trace
          pix_Time_Offset = eye->GetMirrorTimeOffset(pix_tel_ID);
          pix_Bin_Width   = tel_Geometry.GetFADCBinning(); // in units of 100ns i think
          pix_Bin_Offset  = int(pix_Time_Offset/pix_Bin_Width); // in units of 100ns bins
          

          
          // Process the pixel trigger from rec

          if (pix_Status == 4) {
            // Pixel has a pulse and is not noise triggered
            pix_Pulse_Start = RecPixel.GetPulseStart(iPix);
            pix_Pulse_Stop  = RecPixel.GetPulseStop (iPix);

          } else {
            // Pixel has no pulse, so we set the start and stop to 0
            pix_Pulse_Start = 0;
            pix_Pulse_Stop  = 0;
          }

          for (unsigned int t =0; t < pix_trace_length; t++) {
            if (t >= pix_Pulse_Start && t <= pix_Pulse_Stop && pix_Pulse_Start != pix_Pulse_Stop) {
              pix_Trigger_array.push_back(true);
            } else {
              pix_Trigger_array.push_back(false);
            }
          } 

          // Shift the trace, and trigger arrays by the bin offset - adding -999 to the start of the trace
          if (pix_Bin_Offset > 0) {
            for (int i_offset = 0; i_offset < pix_Bin_Offset; i_offset++) {
              pix_Trace.insert(pix_Trace.begin(), -999); // Add -999 to the start of the trace
              pix_Trigger_array.insert(pix_Trigger_array.begin(), false); // Add false to the start of the trigger array
            }
          } // Only +ve offsets possible

          Rec_Trigger_array.push_back(pix_Trigger_array);
          pix_Trigger_array.clear();

          // Store the data
          Signal_array.push_back(pix_Trace);
          Pos_array.push_back(std::make_pair(pix_Theta, pix_Phi));
          ID_array.push_back(pix_ID);
          Status_array.push_back(pix_Status);


        } // Pixel Loop

        if (verbose ) {std::cout << "        Successfully read Event " << NEvent_Scanned << " " << theRecEvent->GetEventId() << " Eye: " << eyeID << std::endl;} 
        
        //  Event checks. 1, check the duration of the Rec Trigger, must be <20 bins
        unsigned int Rec_Trigger_Duration = 0;
        unsigned int t_bin = 0;
        bool Scanned_Full_Array = false;
        unsigned int N_pixels_participating = 0;

        while (!Scanned_Full_Array) {
          N_pixels_participating = 0;

          for (unsigned int i_pix = 0; i_pix < Rec_Trigger_array.size(); i_pix++) {
            if (t_bin < Rec_Trigger_array[i_pix].size()) {
              N_pixels_participating += 1;
              if (Rec_Trigger_array[i_pix][t_bin]) {
                Rec_Trigger_Duration++;
                break; // No need to check other pixels for this time bin
              }
            }
          }

          if (N_pixels_participating == 0) {
            Scanned_Full_Array = true;
            break; // No more pixels to check
          }
          t_bin++;
        }
        
        if (Rec_Trigger_Duration > 20) {
          if (verbose) {std::cout << "Event " << NEvent_Scanned << " has Rec Trigger Duration " << Rec_Trigger_Duration << " bins. Skipping." << std::endl;}
          continue;
        }

        // If got here, event is good for further processing, save to file
        NEvent_Accepted += 1;
        if (verbose) {std::cout << "Event " << NEvent_Scanned << " accepted with Rec Trigger Duration " << Rec_Trigger_Duration << " bins." << std::endl;}

        // Generate file name
        std::string base_filename = "Event_" + theRecEvent->GetEventId() + "_Eye_" + std::to_string(eyeID) + ".csv";
        std::string full_filename = save_file_path + base_filename;
        std::ofstream outFile(full_filename);
        if (!outFile.is_open()) {
          std::cerr << "Error: Could not open file " << full_filename << " for writing." << std::endl;
          return 1;
        }

        // Write the metadata to file
        outFile << "# EventID: " << theRecEvent->GetEventId() << std::endl;
        outFile << "# EyeID: " << eyeID << std::endl;
        outFile << "# TotalPixels: " << Signal_array.size() << std::endl;
        // Next write the Status and Position arrays
        outFile << "# PixelID, Status, Theta, Phi" << std::endl;
        for (unsigned int i = 0; i < Signal_array.size(); i++) {
          outFile << ID_array[i] << ", " << Status_array[i] << ", " << Pos_array[i].first << ", " << Pos_array[i].second << std::endl;
        }
        // Now write the signal array - each row is a pixel, columns are time bins
        outFile << "# Signal Array (rows: pixels, columns: time bins)" << std::endl;
        for (const auto& trace : Signal_array) {
          for (size_t j = 0; j < trace.size(); j++) {
            outFile << trace[j] << ",";
          }
          outFile << std::endl;
        }
        // Now write the Rec Trigger array - each row is a pixel, columns are time bins
        outFile << "# Rec Trigger Array (rows: pixels, columns: time bins)" << std::endl;
        // To Save space, only write the pixels that have trigger, otherwise replace the row with a single zero
        for (const auto& trigger_row : Rec_Trigger_array) {
          bool has_trigger = false;
          for (bool val : trigger_row) {
            if (val) {
              has_trigger = true;
              break;
            }
          }
          
          if (!has_trigger) {
            outFile << "0" << std::endl;
            continue;
          }

          for (size_t j = 0; j < trigger_row.size(); j++) {
            outFile << trigger_row[j] << ",";
          }
          outFile << std::endl;
        }

        outFile.close();
        if (verbose) {std::cout << "Event " << NEvent_Scanned << " data written to " << full_filename << std::endl;}


        
      } // Eye Loop
    } // Event Loop

    if (verbose) {std::cout << "Finished processing file: " << file << std::endl;}
    delete theRecEvent;
    delete det_Geometry;
  } // File Loop

  std::cout << "All files processed successfully, with total of " << NEvent_Scanned << " events and " <<NEyes_Scanned << " eyes scanned and " << NEvent_Accepted << " events accepted." << std::endl;
  
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

