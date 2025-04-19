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
  vector<string> filenames;
  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
      filenames.push_back(arg); // Treat as a filename
  }
  
  
  std::ofstream AllValidThetas("AllValidThetas.csv");
  std::ofstream AllValidPhis("AllValidPhis.csv");
  


  
  for (const string& file : filenames) {
  
    RecEventFile dataFile(file.c_str());
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);
    EyeGeometry & eyeGeometry = detGeometry -> GetEye(5); // HEAT

    // I need to go over all possible Pixels in the three possible telescopes
    for (unsigned int TelID = 1; TelID <4; TelID++) { 
      TelescopeGeometry & TelGeometry = eyeGeometry.GetTelescope(TelID);
      for (unsigned int PixelID = 0; PixelID < 1000; PixelID++) { // 1000 is the max number of pixels in a telescope
        // Get the pixel geometry
        double Phi   = TelGeometry.GetPixelPhi(PixelID,"upward");
        double Theta = TelGeometry.GetPixelOmega(PixelID,"upward");
        cout << "PixelID: " << PixelID << " TelID: " << TelID << " Theta: " << Theta << " Phi: " << Phi << endl;
      } //Pixel loop
    }//Tel loop
  }//File loop
}// Main Exit

        // } else { // No Pulse, So we need to do some shenanegans. 
        //   //Pixel Metainfo
        //   PixelID = RecPixel.GetID(iPix);
        //   TelID   = RecPixel.GetTelescopeId(iPix);
        //   EyeID   = 5; // Explicitly is 5, for this analysis
        //   // Status duh
        //   // Geometry Info
        //   EyeGeometry       & HEGeometry  = detGeometry -> GetEye(EyeID);
        //   TelescopeGeometry & TelGeometry = HEGeometry.GetTelescope(TelID);
        //   Phi   = TelGeometry.GetPixelPhi(PixelID,"upward");
        //   Theta = TelGeometry.GetPixelOmega(PixelID,"upward");
        //   TimeOffset  = 0; // -------------------------------------------------------------------- Not actually what it should be, Check With Reference on how to get this

        //   // Signal Info -> This is where ths gets interesting
        //   // ----> We gotta decide what is the min and max time where a pulse exists on the neighbouring pixels.
        //   // ----> Basically loop over all other pixels, and if they fall into the first category, iteratively make window of Trace Collection larger
        //   // ----> If there is no pixel close enough, than we discard this pixel and go onto the next one i the main pixel loop.
        //   // ----> The above check for SDP Fit pixels instead of just pulsed pixels allows us to not care about the noize triggered pulses.
          
        //   ////////// Pulse Finding 
        //   PulseStart = 200000;
        //   PulseStop  = 0;

        //   for (unsigned int jPix = 0; jPix < Total_Pixels_inEvent; jPix++) {
        //     unsigned int jPix_Status = RecPixel.GetStatus(jPix);
        //     if (jPix_Status >= 2) { // Found a Pulse pixel
        //       unsigned int jPix_PixelID = RecPixel.GetID(jPix);
        //       double jPix_Theta = TelGeometry.GetPixelOmega(jPix_PixelID,"upward"); // Dont have to recall new Geometry objects
        //       double jPix_Phi   = TelGeometry.GetPixelPhi(jPix_PixelID,"upward");  // Guaranteed to be in the same telescope for this analysis
        //       // Check if the pixel is close enough to be considered a neighbour
        //       if (keep_pixel(Theta, Phi, jPix_Theta, jPix_Phi, 2.0)) { // 2 degrees
        //         double jPix_PulseStart = RecPixel.GetPulseStart(jPix);
        //         double jPix_PulseStop  = RecPixel.GetPulseStop(jPix);
        //         if (jPix_PulseStart < PulseStart) {
        //           PulseStart = jPix_PulseStart;
        //         }
        //         if (jPix_PulseStop > PulseStop) {
        //           PulseStop = jPix_PulseStop;
        //         }
        //       } else {
        //         // Do nothing, not a neighbour
        //       }
        //     }
        //   } // jPix Loop
        //   // Found Pulse I Suppose
    