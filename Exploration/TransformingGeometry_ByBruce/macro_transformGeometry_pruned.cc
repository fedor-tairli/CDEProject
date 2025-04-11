// transform Gen geometries (e.g. SDP) from HECO reference to HE reference.
//  BRD 13 March 2025
// example input file
// /remote/tychodata/ftairli/data/CDE/MC/low/b01/PCGF/ADST.PCGF.3000000*



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


// Function to rotate a vector from siteCS to eyeCS
TVector3 RotateVectorToMyCS(const TVector3& v_siteCS, const TVector3& z_eye_siteCS) {
    // Normalize the given z-axis of eyeCS in system siteCS
    TVector3 z_eye_siteCS_unit = z_eye_siteCS.Unit();
    
    // Define the original z-axis of siteCS
    TVector3 z_siteCS(0, 0, 1);
    
    // Compute the rotation axis as z_siteCS � z_eye_siteCS_unit
    TVector3 rotation_axis = z_siteCS.Cross(z_eye_siteCS_unit);
    
    // Compute the rotation angle using dot product
    double rotation_angle = std::acos(z_siteCS.Dot(z_eye_siteCS_unit)); // Angle in radians

    // Rotate the input vector around the computed axis
    TVector3 v_myCS = v_siteCS;
    // BRD change rotation_angle to its negative
    v_myCS.Rotate(-rotation_angle, rotation_axis);
    
    return v_myCS;
}




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


  int nQualityEvents = 0; // just a counter

  // ----------------------------------------------------------------------
  // Loop over files
  // read the file as the second to last argument
  for (int iFile = 1; iFile <= argc - 1; iFile++) {

    RecEventFile dataFile(argv[iFile]);
    RecEvent* theRecEvent = new RecEvent();
    dataFile.SetBuffers(&(theRecEvent));

    DetectorGeometry * detGeometry = new DetectorGeometry();
    dataFile.ReadDetectorGeometry(*detGeometry);

    unsigned int ntotThisFile = dataFile.GetNEvents();

    // ----------------------------------------------------------------------
    // loop over all the events in the file
    for (unsigned int iEvent = 0; iEvent < ntotThisFile; iEvent++) {
	  //  for (unsigned int iEvent = 0; iEvent < 10000; iEvent++) {

      // if there are no more events left in this file, move on the the next file
      if ((!dataFile.ReadEvent(iEvent)) == RecEventFile::eSuccess) continue;


      // if(iEvent%1000 ==0) cout << iEvent << endl;

      const string auger_id = theRecEvent->GetEventId();
      const SDEvent& sevent = theRecEvent->GetSDEvent();
      const SdRecShower &recSDShower = sevent.GetSdRecShower();
      const int eventID = sevent.GetEventId();
      const long int gpssecond = sevent.GetGPSSecond();

      // some shower gen parameters
      // const double genEcal = theRecEvent->GetGenShower().GetCalorimetricEnergy();
      const double genE = theRecEvent->GetGenShower().GetEnergy();
      const int primary = theRecEvent->GetGenShower().GetPrimary();
      TVector3 coreSiteCS = theRecEvent->GetGenShower().GetCoreSiteCS();
      TVector3 axisCoreCS = theRecEvent->GetGenShower().GetAxisCoreCS();

      //From FTelescopeList.  Apeear to be different to angles stored in DetectorGeometry i.e
      //CO backwall angle:  Detector Geom 243.223 FTelescopeList 243.022
      //HE backwall angle:  Detector Geom 273.2 FTelescopeList 273.0
      double CObackwallAngle = 243.0219*kPi/180.;
      double HEbackwallAngle = 273.0*kPi/180.;

      double HECO_Rp = 0;
      double HECO_SDPPhi = 0;
      double HECO_SDPTheta = 0;

      int eye5flag = 0;
      int eye6flag = 0;

      double proj_axis_error = 0;
      double core_dist_horiz = 0;
      TVector3 my_axis_eye5, my_axis_eyeCS, my_axis_myCS;
      TVector3 my_sdp_eyeCS;
      double sdpTheta_eye5, sdpPhi_eye5, sdpTheta_eye6, sdpPhi_eye6;
      double sdp_space_angle_eye6 =0;
      // ----------------------------------------------------------------------
      // Loop over eyes
      std::vector<FDEvent> & fdEvents = theRecEvent->GetFDEvents();
      for (vector<FDEvent>::iterator eye = fdEvents.begin(); eye != fdEvents.end(); ++eye) {  

        const unsigned int eyeID=eye->GetEyeId();

        const EyeGeometry& eye_geom = (*detGeometry).GetEye(eyeID);
        TVector3 eyepos = eye_geom.GetEyePos();  // in site CS
        TVector3 eyeCore_siteCS = coreSiteCS - eyepos;
        TVector3 coreAtEyeAltitude = theRecEvent->GetGenShower().GetCoreAtAltitudeSiteCS(eyepos(2));
        TVector3 eyeCoreAtEyeAltitude = coreAtEyeAltitude - eyepos;
            
        const FdRecGeometry & recGeometry = eye->GetFdRecGeometry();
        const FdGenGeometry & genGeometry = eye->GetGenGeometry();
        const FdRecShower   & theShower   = eye->GetFdRecShower();

        const double Chi0     = recGeometry.GetChi0()*180./kPi;
        const double Rp       = recGeometry.GetRp()/1000.;
        const double SDPtheta = recGeometry.GetSDPTheta()*180./kPi;
        const double SDPphi   = recGeometry.GetSDPPhi()*180./kPi;

        const double genChi0     = genGeometry.GetChi0()*180./kPi;
        const double genRp       = genGeometry.GetRp();
        const double genSDPtheta = genGeometry.GetSDPTheta()*180./kPi;
        const double genSDPphi   = genGeometry.GetSDPPhi()*180./kPi;
        TVector3 sdp = genGeometry.GetSDP();
        const double CoreEyeDistance    = recGeometry.GetCoreEyeDistance()/1000.;
        const double genCoreEyeDistance = genGeometry.GetCoreEyeDistance()/1000.;

        


        if (eyeID == 6){
          eye6flag = 1;
          HECO_Rp       = genRp;
          HECO_SDPPhi   = genSDPphi;
          HECO_SDPTheta = genSDPtheta;

          //will use four coordinate systems.  eyeCS, coreCS and siteCS as
          //usual.  Another CS, called myCS is the same as the eyeCS,
          //except that the (x,y) axes point (east, north).

          // calculate axis parameters from chi0.
          TVector3 eyevertical;
          eyevertical.SetXYZ(0,0,1);

          TVector3 horizontalInSDP = sdp.Cross(eyevertical);
          my_axis_eyeCS = horizontalInSDP.Unit();
          my_axis_eyeCS.Rotate(-genChi0*kPi/180., sdp); // this is in the eye CS

          double backwallangle = CObackwallAngle ;
          my_axis_myCS = my_axis_eyeCS; 
          // my_axis_myCS.Rotate(backwallangle,eyevertical); // in a cartesian CS, with vertical at eye = (0,0,1) and x as east and y as north.
          my_axis_myCS.RotateZ(backwallangle); // in a cartesian CS, with vertical at eye = (0,0,1) and x as east and y as north.
          my_axis_myCS = my_axis_myCS.Unit(); 
          cout << "my_axis_myCS " << my_axis_myCS(0) << " " << my_axis_myCS(1) << " " << my_axis_myCS(2) << endl;
          cout << "axisCoreCS   " << axisCoreCS(0) << " " << axisCoreCS(1) << " " << axisCoreCS(2) << endl;

          
          //THE WINNER!!
          TVector3 eye_z_siteCS;
          eye_z_siteCS.SetXYZ(1.,2.,3.);
          eye_z_siteCS.SetTheta(eye_geom.GetEyeThetaZ());
          eye_z_siteCS.SetPhi(eye_geom.GetEyePhiZ());
          eye_z_siteCS.SetMag(1);
          
          TVector3 eyeCore_myCS = RotateVectorToMyCS(eyeCore_siteCS, eye_z_siteCS);
          backwallangle = eye_geom.GetBackWallAngle();
          TVector3 eyeCore_eyeCS = eyeCore_myCS;
          eyeCore_eyeCS.RotateZ(-backwallangle);

          my_sdp_eyeCS = my_axis_eyeCS.Cross(eyeCore_eyeCS);
          my_sdp_eyeCS = my_sdp_eyeCS.Unit();

          TVector3 genSDP = eye -> GetGenGeometry().GetSDP();
          sdp_space_angle_eye6 = my_sdp_eyeCS.Angle(genSDP)*180./kPi;
          cout << "space angle error of sdp " << sdp_space_angle_eye6 << " *******" <<  endl;


        }

        nQualityEvents++;

      } // eyes
      

      if (eye6flag == 1){

      }


    } // events in file
    
  } // files

  printf("Complete.\n");
  
  //now works after fixing call to TApplication above
  //theApp.Run();
  
  return 0;
}
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


