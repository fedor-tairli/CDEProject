// C++
#include <iostream> // required for cout etc
#include <fstream>  // for reading in from / writing to an external file
#include <string>
#include <vector>
#include <cmath>
#include <ctime>  // For time functions
#include <getopt.h> // For parsing command-line arguments
//#include <filesystem> // For file handling file paths etc.
#include <cstdlib>

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

//using namespace std;


// Functions Defined at the bottom - Help with data Parsing, and usage
std::string getCurrentDateTime();

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
  
  // Variables for command-line arguments
  std::string InputFileName, OutputFileName;
  int n = 1000000; // Default value for -n
  int minStatus = 0; // Default value for --minStatus

  // Define long options for getopt
  static struct option longOptions[] = {
      {"minStatus", required_argument, nullptr, 0},
      {nullptr, 0, nullptr, 0}
  };

  int opt;
  int optionIndex = 0;

  // Parse command-line arguments
  while ((opt = getopt_long(argc, argv, "n:", longOptions, &optionIndex)) != -1) {
      switch (opt) {
          case 'n':
              n = std::stoi(optarg); // Parse the number after -n
              break;
          case 0:
              if (std::string(longOptions[optionIndex].name) == "minStatus") {
                  minStatus = std::stoi(optarg); // Parse the number after --minStatus
              }
              break;
          default:
              std::cerr << "Invalid argument provided." << std::endl;
              exit(1);
      }
  }

  // Check remaining arguments for input and output file names
  if (optind + 2 != argc) {
      std::cerr << "Error in input arguments. Expected 2 file names (input and output)." << std::endl;
      exit(1);
  }

  InputFileName = argv[optind];     // First remaining argument is the input file
  OutputFileName = argv[optind + 1]; // Second remaining argument is the output file

  // Check that the OutputFile can be made
  //std::filesystem::path outputPath(OutputFileName);
  //std::filesystem::create_directories(outputPath.parent_path());
  std::string command = "mkdir -p " + OutputFileName.substr(0,OutputFileName.find_last_of('/'));
  system(command.c_str());
  // Print parsed arguments for verification
  std::cout << "Begin Execution at " << getCurrentDateTime() << std::endl;
  std::cout << "Input File       : " << InputFileName  << std::endl;
  std::cout << "Output File      : " << OutputFileName << std::endl;
  std::cout << "Flag -n          : " << n              << std::endl;
  std::cout << "Flag --minStatus : " << minStatus      << std::endl;

  // Cleanout the output file if it exists
  std::ofstream OutputFile(OutputFileName.c_str());
  if (!OutputFile.is_open()) {
    std::cerr << "Error opening output file: " << OutputFileName << std::endl;
    exit(1);
  }

  // ------------------------------------------------------
  // Reading traces
  // Run the loop over files and events
  unsigned int NEvent  = 0;
  unsigned int NTraces = 0;

  RecEventFile dataFile(InputFileName.c_str());
  RecEvent* theRecEvent = new RecEvent;
  dataFile.SetBuffers(&(theRecEvent));

  unsigned int ntotThisFile = dataFile.GetNEvents();
  for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
    if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue; // Move to next file if no more events
      // Loop over the FDEvents
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      for (std::vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {
        
        const unsigned int eyeID=eye->GetEyeId();
        if (eyeID <5) { continue; }// Check that no HEAT is present (eyeID <5)
        
        NEvent++;  // Only do that after the Eye values have been read (dont want to add everything to even level data and then have the eye fail)
        
        // Preallocate the variables for data
        unsigned int PulseStart ;
        unsigned int PulseStop  ;
        unsigned int Status     ;

        std::vector<double> TraceBins(1000,0.0);

        // Now we go through the pixels. 
        // Writing into file will be done in the loop, so not necessaty to make a new scope.
        // Here we write the data to the file
        FdRecPixel & RecPixel = eye->GetFdRecPixel();
        unsigned int Total_Pixels_inEvent   = RecPixel.GetNumberOfPixels();
        // unsigned int Total_Pulsed_Pixels    = RecPixel.GetNumberOfSDPFitPixels(); // Should USe GetNumberOfPulsedPixels(); but can cut some shit events by using this
        if (Total_Pixels_inEvent == 0) { continue; }
      
        // Now we go through the pixels. 

        for (unsigned int iPix = 0; iPix < Total_Pixels_inEvent; iPix++) {
          Status = RecPixel.GetStatus(iPix);
          //Now we hav two cases here - Pulse and No Pulse
          if (Status < minStatus) {continue;}
          if (Status > 1 ) { // Pulse was already detected by the regular trigger
            PulseStart = RecPixel.GetPulseStart(iPix);
            PulseStop  = RecPixel.GetPulseStop(iPix);
            TraceBins  = RecPixel.GetTrace(iPix);
          
        
          } else { // No Pulse trigger happened, so we dont get pulse bins
            PulseStart = 0;
            PulseStop  = 0;
            TraceBins = RecPixel.GetTrace(iPix);  
          }
        
          OutputFile << Status     << ",";
          OutputFile << PulseStart << ",";
          OutputFile << PulseStop  << ",";
        
          for (unsigned int iTraceBin = 0; iTraceBin < 1000; iTraceBin++) {
            OutputFile << TraceBins[iTraceBin] << ",";
          } // Trace Storage Loop
          OutputFile << std::endl; // End of pixel row 
          NTraces++; // Go to next row when pixel was written.
          if (NTraces>=n) {
            std::cout << "Finished Reading with a total of " << NEvent  << "Eyes   " << std::endl;
            std::cout << "Reached limit of " << n << " traces. Stopping." << std::endl;
            OutputFile.close();
            delete theRecEvent;
            return 0; // Exit after reaching the limit
          }
        } // Pixel Loop
      
      } // Eye Loop
    } // Event Loop
  delete theRecEvent;
  std::cout << "Finished Reading with a total of " << NEvent  << "Eyes   " << std::endl;
  std::cout << "Finished Reading with a total of " << NTraces << "Traces " << std::endl;

  return 0;
  // No File Loop
} // Main Exit




// ----------------------------------------------------------------------
std::string getCurrentDateTime() {
    // Get the current time
    std::time_t now = std::time(nullptr);
    std::tm* localTime = std::localtime(&now);

    // Format the time into a string
    std::ostringstream dateTimeStream;
    dateTimeStream << (1900 + localTime->tm_year) << "-"  // Year
                   << (localTime->tm_mon + 1) << "-"      // Month
                   << localTime->tm_mday << " "           // Day
                   << localTime->tm_hour << ":"           // Hour
                   << localTime->tm_min << ":"            // Minute
                   << localTime->tm_sec;                 // Second

    return dateTimeStream.str();
}
