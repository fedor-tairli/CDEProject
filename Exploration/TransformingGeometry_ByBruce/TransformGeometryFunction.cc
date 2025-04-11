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
//  Coordinates of the observatory

double HE_BackwallAngle_1 = 273.2   *kPi/180.;
double HE_BackwallAngle_2 = 273.0   *kPi/180.;
double CO_BackwallAngle_1 = 243.223 *kPi/180.;
double CO_BackwallAngle_2 = 243.0219*kPi/180.;


TVector3 HE_Position = TVector3(-31741.12427975,  15095.57420328,    210.54774754);
TVector3 CO_Position = TVector3(-31895.75932067,  15026.12801882,    214.90194976);

double HE_EyeThetaZ = 0.005507726051748029;
double HE_EyePhiZ   = 2.6959186109405837;
double CO_EyeThetaZ = 0.00552489736362741;
double CO_EyePhiZ   = 2.69959138256139;

TVector3 HE_EyeZ(){
    TVector3 eye_z;
    eye_z.SetXYZ(1., 2., 3.);
    eye_z.SetTheta(HE_EyeThetaZ);
    eye_z.SetPhi(HE_EyePhiZ);
    eye_z.SetMag(1.0);
    return eye_z;
}

TVector3 CO_EyeZ(){
    TVector3 eye_z;
    eye_z.SetXYZ(1., 2., 3.);
    eye_z.SetTheta(CO_EyeThetaZ);
    eye_z.SetPhi(CO_EyePhiZ);
    eye_z.SetMag(1.0);
    return eye_z;
}

// ----------------------------------------------------------------------
TVector3 RotateVectorToEyeCS(const TVector3& v_in_siteCS, const TVector3& EyeZ_in_siteCS) {
    // Make copies of the input vectors and set up the reference to rotate to
    TVector3 EyeZ_siteCS = EyeZ_in_siteCS.Unit();
    TVector3 v_siteCS    = v_in_siteCS;
    TVector3 Z_siteCS(0, 0, 1);
    
    // Compute the rotation axis and rotation angle
    TVector3 rotation_axis  = Z_siteCS.Cross(EyeZ_siteCS);
    double   rotation_angle = std::acos(Z_siteCS.Dot(EyeZ_siteCS)); // Angle in radians
    v_siteCS.Rotate(-rotation_angle, rotation_axis);
    
    return v_siteCS;
}


// ----------------------------------------------------------------------

TVector3 SDP_in_eyeCS(const TVector3 &axis_coreCS, const TVector3 &core_siteCS, double EyeBackwallAngle_1, double EyeBackwallAngle_2,
                      const TVector3 &EyePosition, const TVector3 &EyeZ) {
    
    // Eye position and vertical in siteCS
    
    TVector3 eye_to_Core_siteCS = core_siteCS - EyePosition;
    TVector3 eye_to_Core_eyeCS = RotateVectorToEyeCS(eye_to_Core_siteCS, EyeZ);
    eye_to_Core_eyeCS.RotateZ(-EyeBackwallAngle_1);

        // This operation gives my_axis_myCS == axis_coreCS    
        //   double backwallangle = CObackwallAngle ;
        //   my_axis_myCS = my_axis_eyeCS; 
        //   my_axis_myCS.Rotate(backwallangle,eyevertical); // in a cartesian CS, with vertical at eye = (0,0,1) and x as east and y as north.
        //   // my_axis_myCS.RotateZ(backwallangle); // in a cartesian CS, with vertical at eye = (0,0,1) and x as east and y as north.
        //   my_axis_myCS = my_axis_myCS.Unit(); 
        //   cout << "my_axis_myCS " << my_axis_myCS(0) << " " << my_axis_myCS(1) << " " << my_axis_myCS(2) << endl;
        //   cout << "axis_coreCS " << axis_coreCS(0) << " " << axis_coreCS(1) << " " << axis_coreCS(2) << endl;

    TVector3 axis_eyeCS = axis_coreCS;
    axis_eyeCS.RotateZ(-EyeBackwallAngle_2);
    axis_eyeCS = axis_eyeCS.Unit();
    TVector3 sdp_eyeCS = axis_eyeCS.Cross(eye_to_Core_eyeCS);
    return sdp_eyeCS.Unit();
}

// ----------------------------------------------------------------------

//  Test the function

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
  //  gROOT->Reset();  // had to comment out seg fault on linux!
  gROOT->SetStyle("Plain");
  // This makes all plots drawn have X and Y grid lines.
  //gStyle->SetPadGridX(true);
  //gStyle->SetPadGridY(true);

  gStyle->SetLineWidth(1);
  const int canvasX = 800; // pixels
  const int canvasY = 600;




  unsigned int NEvent = 0;
  for (int iFile = 1; iFile <= argc-1; iFile++) {
    RecEventFile dataFile(argv[iFile]);
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();
    // cout << "reading file " << dataFile.GetActiveFileName() << " with " << ntotThisFile << " events." << endl;
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
        if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue;

        NEvent++;
        TVector3 core_siteCS = theRecEvent->GetGenShower().GetCoreSiteCS();
        TVector3 axis_coreCS = theRecEvent->GetGenShower().GetAxisCoreCS();

        //Eye Loop, Find HEAT
        std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
        // cout << "Number of FDEvents: " << &fdEvents << endl;
        for (vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {  
            const unsigned int eyeID=eye->GetEyeId();
            if (eyeID != 5 && eyeID != 6) continue; // Only process HE and CO
            const EyeGeometry& eye_geom = (*detGeometry).GetEye(eyeID);


            TVector3 SDP_Calculated;
            if (eyeID == 5) {SDP_Calculated = SDP_in_eyeCS(axis_coreCS, core_siteCS, HE_BackwallAngle_1,HE_BackwallAngle_2, HE_Position, HE_EyeZ());}
            if (eyeID == 6) {SDP_Calculated = SDP_in_eyeCS(axis_coreCS, core_siteCS, CO_BackwallAngle_1,CO_BackwallAngle_2, CO_Position, CO_EyeZ());}
            
            TVector3 SDP_Gen = eye ->GetGenGeometry().GetSDP();
            SDP_Gen = SDP_Gen.Unit();
            double sdp_space_angle = SDP_Calculated.Angle(SDP_Gen)*180./kPi;
            cout << "Processing event " << NEvent << " With Eye ID " << eyeID ;
            cout << ", SDP Space Angle: " << sdp_space_angle << endl;           
            
        } // Eye iteration
    } // Events iteration
  } //Files iteration
} // main 

// int
// main(int argc, char **argv)
// {
//   // Check if the user specified what files to analyse
//   if (argc < 2) {
//     usage();
//     return 1;
//   }

//   //Modify call (thanks Violet) to avoid seg fault in newer Offline
//   //  TApplication theApp("app", &argc, argv);  
//   TApplication theApp("app", NULL, NULL);  
//   // Makes your plots look "normal". The default background isn't plain white.
//   //  gROOT->Reset();  // had to comment out seg fault on linux!
//   gROOT->SetStyle("Plain");
//   // This makes all plots drawn have X and Y grid lines.
//   //gStyle->SetPadGridX(true);
//   //gStyle->SetPadGridY(true);

//   gStyle->SetLineWidth(1);
//   const int canvasX = 800; // pixels
//   const int canvasY = 600;




//   unsigned int NEvent = 0;
//   for (int iFile = 1; iFile <= argc-1; iFile++) {
//     RecEventFile dataFile(argv[iFile]);
//     RecEvent* theRecEvent = new RecEvent();
//     dataFile.SetBuffers(&(theRecEvent));

//     DetectorGeometry * detGeometry = new DetectorGeometry();
//     dataFile.ReadDetectorGeometry(*detGeometry);

//     unsigned int ntotThisFile = dataFile.GetNEvents();
//     // cout << "reading file " << dataFile.GetActiveFileName() << " with " << ntotThisFile << " events." << endl;
//     for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++){
//         if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue;

//         NEvent++;
//         TVector3 coreSiteCS = theRecEvent->GetGenShower().GetCoreSiteCS();
//         TVector3 axis_coreCS = theRecEvent->GetGenShower().GetAxisCoreCS();


//         //Eye Loop, Find HEAT
//         std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
//         // cout << "Number of FDEvents: " << &fdEvents << endl;
//         for (vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {  
//             const unsigned int eyeID=eye->GetEyeId();
//             if (eyeID != 5 && eyeID != 6) continue; // Only process HE and CO
            
//             TVector3 Eye_Position;
//             TVector3 EyeZ;
//             double Eye_BackwallAngle;
//             if (eyeID == 5) {
//                 Eye_Position = HE_Position;
//                 EyeZ = HE_EyeZ();
//                 Eye_BackwallAngle = HE_BackwallAngle;
//             }
//             if (eyeID == 6) {
//                 Eye_Position = CO_Position;
//                 EyeZ = CO_EyeZ();
//                 Eye_BackwallAngle = CO_BackwallAngle;
//             }

//             TVector3 SDP_Gen = eye -> GetGenGeometry().GetSDP();
//             double genChi0   = eye -> GetGenGeometry().GetChi0()*180./kPi;
            
//             //  Do calculation now
//             TVector3 eye_to_Core_siteCS = coreSiteCS - Eye_Position;
//             TVector3 eye_to_Core_eyeCS = RotateVectorToEyeCS(eye_to_Core_siteCS, EyeZ);
//             eye_to_Core_eyeCS.RotateZ(-Eye_BackwallAngle);

//             // Try to match Axis in CoreCS to the axis in EyeCS
//             // TVector3 axis_eyeCS = axis_coreCS;
//             // axis_eyeCS.RotateZ(-Eye_BackwallAngle);
//             // axis_eyeCS = axis_eyeCS.Unit();

//             // TVector3 expected_axis_eyeCS = SDP_Gen.Cross(TVector3(0, 0, 1));
//             // expected_axis_eyeCS.Unit();
//             // expected_axis_eyeCS.Rotate(-genChi0*kPi/180., SDP_Gen);

//             // cout << "EyeID: " << eyeID << " Expected Axis: " << expected_axis_eyeCS(0) << " " << expected_axis_eyeCS(1) << " " << expected_axis_eyeCS(2) << endl;
//             // cout << "EyeID: " << eyeID << "          Axis: " << axis_eyeCS(0) << " " << axis_eyeCS(1) << " " << axis_eyeCS(2) << endl;
//             //  Instead ty matching the axis in EyeCS to the axis in CoreCS

//             TVector3 axis_from_eyeCS;
//             axis_from_eyeCS = SDP_Gen.Cross(TVector3(0, 0, 1));
//             axis_from_eyeCS = axis_from_eyeCS.Unit();
//             axis_from_eyeCS.Rotate(-genChi0*kPi/180., SDP_Gen);
//             // axis_from_eyeCS.RotateZ(Eye_BackwallAngle);
//             axis_coreCS.RotateZ(-Eye_BackwallAngle);
//             cout << "EyeID: " << eyeID << " Expected Axis: " << axis_coreCS(0) << " " << axis_coreCS(1) << " " << axis_coreCS(2) << endl;
//             cout << "EyeID: " << eyeID << "          Axis: " << axis_from_eyeCS(0) << " " << axis_from_eyeCS(1) << " " << axis_from_eyeCS(2) << endl;


//             // double sdp_space_angle = SDP_Calculated.Angle(SDP_Gen)*180./kPi;
//             // cout << "Processing event " << NEvent << " With Eye ID " << eyeID ;
//             // cout << ", SDP Space Angle: " << sdp_space_angle << endl;           
            
//         } // Eye iteration
//     } // Events iteration
//   } //Files iteration
// } // main 

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