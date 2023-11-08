#################################################################################################
# @info Example of a script loading multiple CERN ROOT files in a TChain                        #
# @date 23/11/05                                                                                #
#                                                                                               #
# TRACE and VERBOSE levels used for very detailed debugging                                     #
#################################################################################################
import ROOT


# Converts the X/Y position to a strip number between 1-192. By convention, strip #1 is the one with negative X/Y coordinate
def calculateStripNb(pos_mm: float, stripPitch_mm: float, nbStrips: int) -> int:
    """
    Converts the X/Y position to a strip number between 1-192. By convention, strip #1 is the one with negative X/Y coordinate
    
    Parameters
    ----------
        pos_mm (float) : position in mm
        stripPitch_mm (float) : strip pitch in mm
        nbStrips (int) : total number of strips in the sensor
        
    Returns
    -------
        stripNb (int) : strip number between 1-192
    """
    
    return int( (pos_mm + stripPitch_mm*nbStrips/2.0) / (stripPitch_mm*nbStrips) * nbStrips + 1.0 )



path = "data/mkJbs_*.root"
chain = ROOT.TChain("OPT")
print(f"Files loaded: {chain.Add(path)}")
print(f"Entries loaded: {chain.GetEntries()}")