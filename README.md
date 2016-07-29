# Multi-view Body Model
A body model used for people re-identification in a multi-view camera context.

## Working directory organization

`/bin` : contains the executables

`/ds`  : contains the dataset structured as follows

* /pers1
    + c00000.png
    + c00000.skel
    + c00001.png
    + c00001.skel
    + l00000.png
    + l00000.skel
    + l00001.png
    + l00001.skel
    + r00000.png
    + r00000.skel
    + r00001.png
    + r00001.skel
* /pers2
    + ...
* /pers3
    + ... 
  
 
`/res`  : contains the results of the test and execution times

`conf.xml`: is the configuration file, the following example works fine


    <?xml version="1.0"?>
    <opencv_storage>
        <MainPath>"../ds/"</MainPath>
        <PersonNames>
            gianluca_sync marco_sync matteol_sync 
            matteom_sync nicola_sync stefanog_sync stefanom_sync
        </PersonNames>
        <ViewNames>
            c l r
        </ViewNames>
        <NumImages type_id="opencv-matrix">
            <rows>7</rows>
            <cols>1</cols>
            <dt>u</dt>
            <data>
                74 84 68 68 62 69 41
            </data>
        </NumImages>
        <OcclusionSearch>1</OcclusionSearch>
    </opencv_storage>

**Note:** 
- Set the field `<rows>` with the appropriate number of persons
- Set the field `OcclusionSearch` to 1 to activate that functionality 
- `NumImages` contains the total number of images for each person (the one with 0 excluded) 
   in the same order as `PersonNames`
  

## Source files

`main.cpp`: contains the code used to test the model 

`MultiviewBodyModel.h`: contains the class that represents the model and the structures used

`MultiviewBodyModel.cpp`: contains methods definitions

Type `multiviewbodymodel` on the console to see the help with a detailed explanation of the overall 
parameters.