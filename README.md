# isobaric-inspection-jupyter

LC-MS proteomic data is complex, many things may go wrong during sample prep, mass spectrometry analysis and data processing. [Proteome Discoverer](https://www.thermofisher.com/se/en/home/industrial/mass-spectrometry/liquid-chromatography-mass-spectrometry-lc-ms/lc-ms-software/multi-omics-data-analysis/proteome-discoverer-software.html) is a proprietary application suite for complete processing from the raw mass spectrometry files to protein abundances. This repository contains the Jupyter notebook for the inspection of the quality of the data from isobaric labeling-based quantitative proteomic analysis.

Mass spectrometry data is publicly available in the [project PXD007647](https://www.ebi.ac.uk/pride/archive/projects/PXD007647) at PRIDE archive. The data consists of 10 *E. coli* samples combined into one isobaric labeling set. Please check the corresponding [publication by Thulin and Andersson](https://journals.asm.org/doi/full/10.1128/AAC.00612-19) for more information about the study.

Raw files have been processed by Proteome Dsicovere 2.4, the results have been exported as tab-delimited text files using the Results Exporter node ("R-friendly headers" on). The output contains the following files:
```python
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_QuanSpectra.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_SpecializedTraces.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_Proteins.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_PeptideGroups.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_MSMSSpectrumInfo.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_PSMs.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_ResultStatistics.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_ProteinGroups.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_InputFiles.txt
/PD_Out/PXD007647_Reproc_TMT-set-2_8fracs_PrSMs.txt
```

The notebook has been created in Jupyter Lab 3.0.16 in Python 3.8, and tested in Windows 10 and Ubuntu 20.04. 
