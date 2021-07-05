#!/usr/bin/env python
# coding: utf-8

# # Checking the Quality of Isobaric Labeling Proteomics Data

# Mass spectrometry (MS)-based proteomics has been emerging as a powerful variety of the modern high-throughput *omics* technologies. Superficial search on Pubmed yields [tens of thousands papers](https://pubmed.ncbi.nlm.nih.gov/?term=proteomics%5BTitle%2FAbstract%5D+AND+%28%28mass+spectrometry%5BTitle%2FAbstract%5D%29+OR+%28LCMS%5BTitle%2FAbstract%5D%29%29+OR+%28LC-MS%5BTitle%2FAbstract%5D%29) where the MS-based proteomics is mentioned in the title or abstract, let alone the papers where in which is a humble workhorse hidden deep in the Methods section.<br>
# [Isobaric labeling-based proteomics](https://en.wikipedia.org/wiki/Isobaric_labeling#:~:text=Isobaric%20labeling%20is%20a%20mass,heavy%20isotopes%20around%20their%20structure.), including [Tandem mass tags (TMT)](https://pubs.acs.org/doi/10.1021/ac0262560) technology, is an approach for multiplexed relative protein quantification that is based on chemical labeling of peptides with the set of the stable isotope-substituted reagents. For a given peptide, all reagents would yield the same total mass, but the signals with distinct masses (reporter ions) would be produced upon fragmentation during a tandem MS experiment. In 2021, the commercially available technology is offering to measure up to 16 samples simultaneously (for more technical information, see [the manual at the vendor's website](https://www.thermofisher.com/document-connect/document-connect.html?url=https%3A%2F%2Fassets.thermofisher.com%2FTFS-Assets%2FLSG%2Fmanuals%2FMAN0018773_TMTproMassTagLabelingReagentsandKits_UG.pdf&title=VXNlciBHdWlkZTogVE1UcHJvIE1hc3MgVGFnIExhYmVsaW5nIFJlYWdlbnRzIGFuZCBLaXRz)).
# Isobaric proteomic data is complex: proteolytic peptides should be identified, the reporter ions must be quantified, the quantification data should be normalized, peptide-level data should be assembled into the protein quantification results. There are plenty of things that may go wrong, thus the thorough inspection of the quality of the acquired isobaric labeling data. This notebook includes the range of command that help me out with checking the quality of the data. I will not go into the biological interpretation or anything like that, this is the inspection from the technical standpoint for the most part. The commands are adapted to the text output format provided by Proteome Discovere 2.4, but they can be adapted to the output from other data processing solutions.

# <h2>Import the libraries</h2>

# Install the dependencies below via *pip* *install* *package-name* if you do not yet have them on your machine.

# In[1]:


import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

sns.set(font_scale = 1.2)
sns.set_style('whitegrid')
sns.set_palette('muted')


# We could also go ahead and define a few useful functions that we're going to use along the way.

# In[2]:


def get_gene_name(descr):
    try:
        gn = descr.split('GN=')[1]
        gn = gn.split(' ')[0]
    except:
        gn = ''
    return gn

def variability_groupby(df, ratio_columns, accession_col='Accession',
                          use_peptides='all',shared_keyword='Shared', unique_infocol='Quan Info'):
    if use_peptides == 'unique':
        df = df[~df[unique_infocol].isin((shared_keyword,))].copy()
    elif use_peptides == 'all':
        pass
    else:
        df = None
        
    dfN = df[ ([accession_col,] + ratio_columns) ]
    
    dfMeans = dfN.groupby([accession_col]).mean()
    dfDev = dfN.groupby([accession_col]).std()
    dfVar = np.divide(dfDev, dfMeans)
    dfVar = dfVar*100
    dfVar.replace(0, np.nan, inplace=True)
    dfVar = dfVar.round(2)
    
    return dfVar

def pca_on_columns(in_df, plotname, nComp=5, compToPlot=(0,1), figWidth = 8 ):
    """
    Takes an expression table, with samples as columns and proteins in rows.
    Figure width is set by the user,
    figure height is derived automatically based on the scale of the PCs. 
    """
    
    samplenames = list(in_df.columns)
    logdata = in_df.to_numpy()

    X = logdata.transpose()

    pca = PCA(n_components=nComp, svd_solver='arpack')
    principalComponents = pca.fit_transform(X)
    
    print('Variance explained by components:')
    print(pca.explained_variance_ratio_)
    
    pcaLoadings = pd.DataFrame(pca.components_.T,
                               columns=[ f'PC{n+1}' for n in range(nComp) ],
                               index=in_df.index)
    
    pcColnames = []
    for n in range(nComp):
        pcColnames.append( f'PC {n+1} ({pca.explained_variance_ratio_[n]*100:.2f}%)' )

    principalDf = pd.DataFrame(principalComponents,
                               columns = pcColnames)
    principalDf['sample'] = samplenames
    principalDf.set_index('sample', inplace = True)
    
    #Find min and max values on x and y axes in order to detrmine the figsize
    arrX = principalDf[ pcColnames[ compToPlot[0] ] ]
    arrY = principalDf[ pcColnames[ compToPlot[1] ] ]
    spanX = max(arrX) - min(arrX)
    spanY = max(arrY) - min(arrY)
    plotH = round(figWidth * spanY / spanX, 1)
    print(f'Figure width set to {figWidth} and height to {plotH}')
    
    f = plt.figure()
    f.patch.set_facecolor('white')
    ax1 = principalDf.plot.scatter(x=pcColnames[ compToPlot[0] ],
                                   y=pcColnames[ compToPlot[1] ],
                                   color='#213A8F',
                                   s=50, alpha=0.7, figsize=(figWidth, plotH) )
    ax1.axhline(color='grey', alpha=0.4, linestyle='--')
    ax1.axvline(color='grey', alpha=0.4, linestyle='--')
    
    for i, row in principalDf.iloc[:, list(compToPlot) ].iterrows():
        ax1.annotate(i, row,
                     xytext=(10,-5), textcoords='offset points',
                     fontsize=12, color='#213A8F')
    ax1.set_title(plotname, fontsize=16)
    plt.grid(b=None)
    
    #plt.show()

    return principalDf, pcaLoadings

def rename_ratios(dfIn):
    """
    Used for the file QuanSpectra
    """
    renaming_dict = {}
    for c in dfIn.columns:
        if 'Abundances Normalized' in c:
            new_name = c.split(' ')[-1]
            new_name = 'Norm_' + new_name
            renaming_dict[c] = new_name
        elif 'Abundance ' in c:
            new_name = c.split(' ')[-1]
            renaming_dict[c] = new_name
    dfOut = dfIn.rename(renaming_dict,axis='columns')
    return dfOut

def rename_ratios4(dfIn):
    """
    Adapted for Peptide Groups table
    """
    renaming_dict = {}
    for c in dfIn.columns:
        if 'Abundance Ratio' in c:
            new_name = c.split(' ')[-3] + '/' + c.split(' ')[-1]
            renaming_dict[c] = new_name
        elif 'Abundances Grouped' in c:
            new_name = c.split(' ')[-1]
            renaming_dict[c] = new_name
        renaming_dict['Master Protein Accessions'] = 'Accession'
    dfOut = dfIn.rename(renaming_dict,axis='columns')
    return dfOut


# <h2>Take a look at the output files from Proteome Discoverer</h2>

# The LC-MS files that I will use for demonstration are publicly available via PRIDE repository with [identifier PXD007647](https://www.ebi.ac.uk/pride/archive/projects/PXD007647). This proteomic analysis was a part of [the study published by Thulin and Andersson](https://journals.asm.org/doi/full/10.1128/AAC.00612-19) in the journal Antimicrobial Agents and Chemotherapy. The particular batch that we will be looking at consists of 10 <i>E. coli</i> samples, labeled with TMT tags and pre-fractionated into 

# To make our lives easier, I have added the *Results Exporter* node to the consensus workflow in Proteome Discovere 2.4, so that we get a bunch of .TXT output files right in the study directory when the processing is finished. There are not many options in that node, I have marked an option "R-friendly headers" as TRUE.

# The output files are in the sub-directory "PD_Out"

# In[3]:


proteomicTxtFiles = [
    os.path.join('/PD_Out', f) for f in os.listdir(os.getcwd() + '/PD_Out') if f.endswith('.txt')
]
for f in proteomicTxtFiles:
    print(f)


# For convenience, assign the name or a project number to the data set.

# In[11]:


PNUM = 'Ecoli TMT'


# <h2>Get on the files one-by-one</h2>

# <h3>File InputFiles</h3>

# In[8]:


df1 = pd.read_csv(
    os.getcwd() + [ n for n in proteomicTxtFiles if 'InputFiles' in n ][0],
    sep='\t'
)
df1['count'] = [1,] * df1.index
print(df1.shape)
print(df1.columns)
df1.head(3)


# The first file is a Proteome Discoverer .MSF file that stores the intermediate processing results:

# In[9]:


df1['File Name'][0]


# The rest of the files are the LC-MS .RAW files, there are eight:

# In[10]:


df1['File Name'][1:]


# What useful info can we find in the InputFiles table?

# In[11]:


list(df1['Study File ID'])


# File IDs follow a pattern F1.x. In Proteome Discoverer, it suggests that we have **one** TMT labeling set, and all eight files are the **fractions** that all belong to the same TMT set. Eight LC-MS files are essentially treated as one big file during data analysis

# In[12]:


df1['Software Revision'].unique()


# These are the versions of Proteome Discoverer (the former) and of the Orbitrap Tune software on the computer that acquired the data (the latter).

# In[13]:


df1['Instrument Name'].unique()


# The first generation Orbitrap Fusion acquired the LC-MS files

# In[14]:


df1['RT Range in min'].unique()


# We could determine the LC-MS file length in minutes, it will be handy for plotting down the line.

# In[15]:


rtRangeMaxVals = [
    float( x.split(' - ')[-1] ) for x in df1['RT Range in min'].unique() if type(x) == str
]
rtRangeMaxVals


# In[16]:


maxRT = int( max(rtRangeMaxVals) )
maxRT


# <h3>File MSMSSpectrumInfo</h3>

# In[17]:


df2 = pd.read_csv(
    os.getcwd() + [ n for n in proteomicTxtFiles if 'MSMSSpectrumInfo' in n ][0],
    sep='\t'
)
df2['count'] = [1,] * df2.index
print(df2.shape)
print(df2.columns)
df2.head(3)


# For an ease of plotting, let's add a categorical column "Identified Peptide" that have two strings as values: "Yes" or "No"

# In[18]:


yes_or_no = lambda x: 'NoID' if x==0 else 'WithID'
df2['Identified Peptide'] = [
    yes_or_no( row['Number of PSMs'] ) for _,row in df2.iterrows()
]
df2.head(3)


# In[19]:


print(df2['Mass Analyzer'].unique())
print(df2['Activation Type'].unique())


# The MSMS spectra were acquired with the ion trap analyzer with CID activation.

# Let's look at the success rate of peptide identification. A very low number of PSMs per MSMS may signal problems with the sample prep, acquisition or database search.

# In[21]:


fig1, ax1 = plt.subplots()
df2[
    ['Identified Peptide', 'count']
].groupby(
    ['Identified Peptide']
).count().plot(
    kind = 'pie', ax = ax1,
    y = 'count', figsize = (4,4),
    autopct = '%1.1f%%', fontsize = 16
)
fig1.set_facecolor("white")
plt.legend('')
plt.suptitle(f'{PNUM} PSMs per MSMS')


# In[22]:


df2[
    ['Spectrum File', 'Number of PSMs']
].groupby(
    ['Spectrum File']
).mean().plot.barh(figsize=(8,6))
plt.title(f'{PNUM} ID Rate by File')
plt.legend('')
plt.xlabel('AVG PSM per MSMS')
plt.ylabel('')


# The ID rate varies between the .RAW files, which can be useful for the optimization of the pre-fractionation method. We can also spot if there was something wrong with one or more of the files.

# In[23]:


df2[
    ['Spectrum File', 'RT in min']
].hist(
    by='Spectrum File', bins=maxRT, layout=(2,4), figsize=(14,6)
)
plt.suptitle(f'{PNUM} All MSMS Spectra Per Min Per File / Retention time on x-axis')


# These plots show the distribution of all the MSMS spectra over the elution profile. The spectra are acquired rather uniformly over the LC gradient.

# In[24]:


df2[
    df2['Number of PSMs'] > 0
][
    ['Spectrum File', 'RT in min']
].hist(
    by='Spectrum File', bins=maxRT, layout=(2,4), figsize=(14,6),
    color = '#1ca641'
)
plt.suptitle(f'{PNUM} Spectra With ID Per Min Per File / Retention time on x-axis')


# In[25]:


df2[
    ['Isolation Interference in Percent', 'Ion Inject Time in ms']
].hist( bins=50, figsize=(12,3) )
plt.suptitle(f'{PNUM}')


# Isolation interference is the percentage of the ion intensity in the isolation window that does not belong to the "main" precursor in that window.<br>Low values mean less probability of co-isolation of two or more peptides that leads to complicated mixture fragmentation spectra.<br>
# Knowing the distribution of the injection times may be useful for tuning the MS method.

# In[26]:


fig1, ax1 = plt.subplots()
df2[
    ['Precursor Charge', 'count']
].groupby(
    ['Precursor Charge']
).count().plot(
    kind='pie', ax=ax1,
    y='count', figsize=(6,5),
    autopct='%1.1f%%', fontsize=14
)
fig1.set_facecolor("white")
plt.suptitle(f'{PNUM} MSMS by Precursor Charge')
plt.legend('')


# The distribution of all the MSMS spectra by charge. The 2+ precursors are prevalent.

# In[27]:


df2[
    ['Precursor Charge', 'Number of PSMs']
].groupby(
    ['Precursor Charge']
).mean().plot.bar(figsize=(8,3), rot=0)
plt.title(f'{PNUM} ID Rate by Charge')
plt.ylabel('PSM per MSMS')
plt.legend('')


# In[28]:


df2[
    ['Identified Peptide', 'Precursor MHplus in Da']
].hist(
    by='Identified Peptide', bins=100, sharex=True, sharey=False,
    layout=(2,1), figsize=(12,6)
)
plt.suptitle(f'{PNUM} Precursor Mass Distributions')
plt.xlabel('Precursor MH+ (Da)')


# In[29]:


df2[
    ['Identified Peptide', 'Precursor mz in Da']
].hist(
    by='Identified Peptide', bins=50, sharex=True, sharey=False,
    layout=(2,1), figsize=(12,6)
)
plt.suptitle(f'{PNUM} Precursor M/Z Distributions')
plt.xlabel('Precursor m/z')


# Same deal with the m/z of the spectra that lead to peptide identifications

# We could segment our table by m/z, let's say, at 100 m/z intervals, and look at ID rates in each segment

# In[30]:


minMZ = df2[['Precursor mz in Da']].min()
print( f'Min {minMZ}' )
maxMZ = df2[['Precursor mz in Da']].max()
print( f'Max {maxMZ}' )


# Let's split it into intervals of 100, this will help to determine which precursor regions are the most effective in terms of peptide identification.

# In[31]:


min100 = int( np.floor(minMZ / 100) )
max100 = int( np.ceil(maxMZ / 100) )

#Generate the dictionary with m/z tuple and corresponding label
intervals = []
for i in range(min100, max100):
    intervals.append( [ 100*i, (i+1)*100 ] )
#Pad the string with zeros for better sorting on subsequent plots
intervals = {
    (x1, x2): f'{x1:04}-{x2:04}' for x1, x2 in intervals
}

def split_mz(mz, intervals):
    intervalString = ''
    for k in intervals.keys():
        if (mz > k[0]) and (mz <= k[1]):
            intervalString = intervals[k]
    return intervalString

df2['MZ interval'] = [
    split_mz( row['Precursor mz in Da'], intervals ) for _,row in df2.iterrows()
]
df2.head(3)


# In[32]:


fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, figsize=(12, 8))
#PSMs per interval
df2[
    ['Number of PSMs', 'MZ interval']
].groupby(
    ['MZ interval']
).sum().plot.bar(ax=ax1, rot=0, color = '#7E94C0')
ax1.set_ylabel('PSM')
#MSMS without ID per interval
df2[
    df2['Number of PSMs'] == 0
][
    ['Number of PSMs', 'MZ interval']
].groupby(
    ['MZ interval']
).count().plot.bar(ax=ax2, rot=0, color = '#7EC0A9')
ax2.set_ylabel('MSMS Without ID')
#ID rate per interval
df2[
    ['Number of PSMs', 'MZ interval']
].groupby(
    ['MZ interval']
).mean().plot.bar(ax=ax3, rot=0)
ax3.set_ylabel('ID Rate')

plt.suptitle(f'{PNUM} MSMS With and Without Identifications', fontsize = 20)
ax1.legend('')
ax2.legend('')
ax3.legend('')


# In[ ]:





# Let's get a more nuanced picture the picture a bit and add the charges to the mixture.<br>
# This can be informative for the optimization of precursor selection settings.

# In[33]:


df2[
    df2['Precursor Charge'] <= 6
][
    ['Precursor Charge','Number of PSMs', 'MZ interval']
].groupby(
    ['Precursor Charge', 'MZ interval']
).mean().plot.barh(figsize=(12,12))
plt.title(f'{PNUM} ID Rates by Charge and m/z Interval')
plt.xlabel('ID Rate')
plt.legend('')


# <h3>File QuanSpectra</h3>

# In[34]:


df3 = pd.read_csv(
    os.getcwd() + [ n for n in proteomicTxtFiles if 'QuanSpectra' in n ][0],
    sep='\t'
)
print(df3.shape)
print(df3.columns)
df3.head(3)


# In[35]:


df3 = rename_ratios(df3)
#Use the lambda function yes_or_no that we have introduced earlier
df3['Identified Peptide'] = [ yes_or_no( row['Number of PSMs'] ) for _,row in df3.iterrows() ]
df3['Sqrt AVG Reporter SN'] = np.sqrt( df3['Average Reporter SN'] )
df3['Log10 AVG Reporter SN'] = np.log10( df3['Average Reporter SN'].replace(0, np.nan) )
print( [ (i, c) for i, c in enumerate(df3.columns) ] )
df3.head(3)


# In[36]:


df3['File ID'].unique()


# Select the columns with normalized abundance

# In[37]:


df3.iloc[:,17:27].replace(0, np.nan).dropna(axis='rows').head(3)


# Select the columns with non-normalized abundance

# In[38]:


df3.iloc[:,27:37].replace(0, np.nan).dropna(axis='rows').head(3)


# In[39]:


sns.set(font_scale = 1.2)
sns.set_style('whitegrid')

fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 12))

np.log10(
    df3.iloc[
        :, 27:37
    ].replace(0, np.nan).dropna(axis='rows')
).boxplot(
    ax=ax1, notch=True, showmeans=True,
    vert=False,
    boxprops= dict(linewidth=1, color='#1214AD'),
    whiskerprops= dict(color='#1214AD'),
    medianprops= dict(linewidth=2), fontsize=12
)
ax1.set_xlabel('Log10 Reporter SN')
ax2.set_ylabel('')
#Add the box plot after normalization
np.log10(
    df3.iloc[
        :, 17:27
    ].replace(0, np.nan).dropna(axis='rows')
).boxplot(
    ax=ax2, notch=True, showmeans=True,
    vert=False,
    boxprops= dict(linewidth=1, color='#00631C'),
    whiskerprops= dict(color='#00631C'),
    medianprops= dict(linewidth=2), fontsize=12
)
ax2.set_xlabel('Log10 Normalized Reporter SN')
ax2.set_ylabel('')

plt.suptitle(f'{PNUM} Set Reporter SN Before and After Normalization', fontsize = 20)


# In[40]:


normCoeffs = df3.iloc[
    :, 17:27
].replace(0, np.nan).dropna(axis='rows').mean().to_numpy() / df3.iloc[
    :, 27:37
].replace(0, np.nan).dropna(axis='rows').mean().to_numpy()
print(f'{PNUM} normalization coefficients:')
for p in zip( df3.iloc[:,27:37].columns, list( np.round(normCoeffs, 2) ) ):
    print(p)


# See the numeric representation of the normalization. The channel 129C was the weakest, but 1.88 is frankly not that bad.

# In[41]:


df3[ ['Average Reporter SN'] ].describe()


# In[42]:


df3[
    ['Spectrum File', 'Average Reporter SN']
].groupby(
    ['Spectrum File']
).mean().plot.barh(figsize=(8,6))
plt.title(f'{PNUM} Average Reporter SN by File', fontsize='large')
plt.xlabel('Averare Reporter SN', fontsize=14)
plt.ylabel('')
plt.legend('')


# Square root scale helps to compress the S/N range while preserving zero values.<br>
# We are very interested to check how many quan spectra have S/N = 0,<br>
# those would have been lost as a result of log-transformation.

# In[43]:


df3[
    ['Identified Peptide', 'Sqrt AVG Reporter SN']
].hist(
    by='Identified Peptide', bins=40, sharex=True,# sharey=True,# log=True,
    layout=(2,1), xrot=0, figsize=(10,6)
)
plt.suptitle(f'{PNUM} Average SN Distributions')
plt.xlabel('Square Root of AVG Reporter SN')


# Use log10 S/N for hexbin plot, otherwise the zero values can saturate the color map.

# In[45]:


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (12, 4)
hb = ax.hexbin(
    x=df3['RT in min'], y=df3['Log10 AVG Reporter SN'],
    gridsize=int(maxRT/2),
    cmap='coolwarm'
)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Count')
plt.title(f'{PNUM} AVG Reporter SN vs RT')
plt.xlabel('RT in min')
plt.ylabel('Log10 AVG Reporter SN')


# Spectra with S/N = 0 are excluded from the plot due to log-transformation, which puts emphasis on the quantified spectra.

# In[46]:


df3.plot.scatter(
    x='RT in min', y='Sqrt AVG Reporter SN',
    s=0.5,color='navy', alpha=0.2, figsize=(10, 4)
)
plt.title(f'{PNUM} AVG Reporter SN vs RT')


# Weak spectra in the dead volume and at the very end are to be expected.<br>
# Are the week spectra (AVG SN < 5) in our data set mostly concentrated at the beggining and at the end? 

# In[47]:


df3[
    df3['Average Reporter SN'] < 5
]['RT in min'].hist(
    bins=maxRT, figsize=(10, 4)
)
plt.title(f'{PNUM} Distribution of Quan Spectra with S/N < 5')
plt.xlabel('RT (min)')


# In[48]:


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (12, 4)
hb = ax.hexbin(
    x=df3['Precursor mz in Da'], y=df3['Log10 AVG Reporter SN'],
    gridsize=maxRT, cmap='coolwarm'
)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Count')
plt.title(f'{PNUM} AVG Reporter SN vs Precursor m/z')
plt.xlabel('Precursor m/z')
plt.ylabel('Log10 AVG Reporter SN')


# In[49]:


df3[
    ['Precursor Charge', 'Average Reporter SN']
].groupby(
    ['Precursor Charge']
).mean().plot.bar(figsize=(8,3), rot=0)
plt.title(f'{PNUM} Average Reporter S/N by Charge')
plt.ylabel('AVG Reporter SN')
plt.legend('')


# <h3>File PSMs</h3>

# In[50]:


df4 = pd.read_csv(
    os.getcwd() + [ n for n in proteomicTxtFiles if 'PSMs' in n ][0],
    sep='\t'
)
print(df4.shape)
print(df4.columns)
df4.head(3)


# In[51]:


df4.plot.scatter(
    x='MHplus in Da', y='Ions Score', s=1,color='navy', alpha=0.4, figsize=(10, 4)
)
plt.title(f'{PNUM} Mascot Ion Score vs Precursor MH+')
plt.xlabel('Precursor MH+')
plt.ylabel('Mascot Ion Score')


# SPS match is a measure of quantitative interference for MS3 TMT workflows calculated by Proteome Discoverer.<br>
# It takes the number of the MS2 fragments that have been selected and isolated for MS3,<br>
# and calculates the percentage of those MS2 that match to the annotated MS2 fragments of the main identified peptide.<br>
# Thus, high SPS match % should mean that the interference from unknown peptides is low.

# In[52]:


df4[
    ['SPS Mass Matches in Percent', 'Isolation Interference in Percent']
].groupby(
    ['SPS Mass Matches in Percent']
).count().plot.bar(figsize=(8,3), rot=0)
plt.title(f'{PNUM} SPS Match % Counts')
plt.ylabel('Count')
plt.xlabel('SPS Match (%)')
plt.legend('')


# Proteome Dsicoverer calculates the isolation interference for precursors as well.<br>
# This is the percentage of signals within the precursor isolation window that do not belong to the main precursor isotopic envelope.

# In[53]:


df4['Isolation Interference in Percent'].hist(
    bins = 20, figsize=(8, 4)
)
plt.title(f'{PNUM} Distribution of Precursor Isolation Interference')
plt.xlabel('Isolation Interference in %')


# The SPS match and precursor interference values are sort of linked, but not very much:

# In[54]:


df4[
    ['SPS Mass Matches in Percent', 'Isolation Interference in Percent']
].groupby(
    ['SPS Mass Matches in Percent']
).mean().plot.bar(figsize=(10,3), rot=0)
plt.title(f'{PNUM} Mean Precursor Isolation Interference by SPS Match %')
plt.ylabel('Mean Isolation Interference (%)')
plt.xlabel('SPS Match (%)')
plt.legend('')


# Inspect the precursor mass error

# In[55]:


fig, ax = plt.subplots()
plt.rcParams["figure.figsize"] = (12, 4)
hb = ax.hexbin(
    x=df4['RT in min'], y=df4['Delta M in ppm'],
    gridsize = int( maxRT/2 ),
    cmap = 'coolwarm'
)
cb = fig.colorbar(hb, ax=ax)
cb.set_label('Count')
plt.title(f'{PNUM} Mass Error vs RT')
plt.xlabel('Retention time, min')
plt.ylabel('Mass error, ppm')


# In[56]:


df4.plot.scatter(
    x='RT in min', y='Delta M in ppm',
    s=1,color='navy', alpha=0.2, figsize=(10, 4)
)
plt.title(f'{PNUM} Mass Error vs RT')


# <h3>File PeptideGroups</h3>

# In[12]:


df5 = pd.read_csv(
        os.getcwd() + [ n for n in proteomicTxtFiles if 'PeptideGroups' in n ][0],
    sep='\t'
)
df5 = rename_ratios4(df5)
print(df5.shape)
print(df5.columns)
df5.head(3)


# In[13]:


df5['Number of PSMs'].unique()


# Find columns that contain abundance ratios

# In[14]:


df5.iloc[:, 15:24].dropna(axis='rows')


# In[15]:


df5.columns[15:24]


# How many peptides have missing quantitative values?

# In[16]:


peptidesMVFraction = 1 - df5.iloc[:, 15:24].dropna(axis='rows').shape[0] / df5.iloc[:, 15:24].shape[0]
peptidesMVFraction = peptidesMVFraction * 100
f'Peptides with missing quan values: {peptidesMVFraction:.1f}%'


# Multi-batch data sets may have very high prevalence of missing values.<br>
# Single-batch isobaric labeling with too many missing values may raise suspicion.<br>
# One could review the processing parameters, as an example

# Let's look at the variability of the peptide ratios within each protein.<br>
# In the ideal case, we expect peptides within a protein to have very similar intensity profiles.

# In[17]:


dfPeptVar = variability_groupby(
    df5[ (['Accession', ] + list(df5.columns[15:24]) ) ].dropna(axis='rows'),
    ratio_columns = list( df5.columns[15:24] )
)
print(dfPeptVar.head(3))
np.log10(
    dfPeptVar.replace(0, np.nan)
).boxplot(
    figsize=(10,5),
    notch=True, showmeans=True, vert=False,
    medianprops= dict(linewidth=2), fontsize=12
)
plt.title(f'{PNUM} Peptide Variability (%) Within Proteins', fontsize=18)
plt.xlabel('Log10 Peptide Variability (%)', fontsize=12)
#plt.xticks(rotation=45, horizontalalignment='right')


# In[18]:


dfPeptVar.describe()


# Pretty low within-protein-variabilities for peptide ratios, with mean values between 9 and 19%

# We could look at the correlations between the abundance ratios on peptide level.<br>
# Should remain careful while interpreting the correlation:<br>
# as I have described in [my recent blog post](https://towardsdatascience.com/correlation-in-isobaric-labeling-proteomics-926045214f96), the correlation coefficients are highly dependent on the way in which the data has been scaled.

# In[19]:


f = plt.figure(figsize=(7,6))
sns.heatmap(
    np.log2(
        df5.iloc[:, 15:24].dropna(axis='rows')
    ).corr(method='pearson').round(2),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=2, linecolor='white'
)
plt.suptitle(f'{PNUM} Pearson Correlations on Peptide Level',fontsize=18)


# A more nuanced depiction with the pairplots

# In[20]:


sns.pairplot(
    np.log2(
        df5.iloc[:, 15:24].dropna(axis='rows')
    ),
    vars=df5.iloc[:, 15:24].columns,
    diag_kind='kde', kind='scatter', markers=',', height = 1.5,
    plot_kws={'s': 8,  'alpha':0.7, 'color': 'navy'}
)
plt.suptitle(f'{PNUM} Pair Plots on Peptide Level', fontsize=18)


# Let's check the efficiency of the enzymatic digestion

# In[21]:


fig1, ax1 = plt.subplots()
df5[
    ['Number of Missed Cleavages', 'Confidence']
].groupby(
    ['Number of Missed Cleavages']
).count().plot(
    kind='pie', ax=ax1,
    y='Confidence', figsize=(4,4),
    autopct='%1.1f%%',fontsize=16
)
fig1.set_facecolor("white")
ax1.set_ylabel('')
plt.legend('')
plt.suptitle(f'{PNUM} Peptides by Missed Cleavages')


# We could also check the cysteine containing peptides. Cysteine residues are usually modified via reduction and alkylation, and it's good to check that the derivatization has been done properly. If not, the abundances of cysteine-containing peptides may be very low in the samples that have experienced issues.

# Let's mark the derivatized cysteine residues in a separate column.

# In[28]:


df5['Cys_Peptide'] = [
    'Yes' if '[C' in x else 'No' for x in df5['Modifications']
]


# In[31]:


df5[ df5['Cys_Peptide'] == 'Yes' ].head(3)


# In[35]:


np.log2(df5[ df5['Cys_Peptide'] == 'Yes' ].iloc[:, 15:24].dropna(axis='rows'))


# In[41]:


f = plt.figure(figsize=(4,8))
sns.heatmap(
    np.log2(
        df5[
            df5['Cys_Peptide'] == 'Yes'
        ].iloc[
            :, 15:24
        ].dropna(axis='rows')
    ),
    square=False, cmap='coolwarm', vmin=-2, vmax=2,
    annot=False, cbar=True
)
plt.suptitle(f'{PNUM} Modified Cys Peptide Log2 Relative Ints',fontsize=14)


# In[46]:


sns.clustermap(
    np.log2(
        df5[
            df5['Cys_Peptide'] == 'Yes'
        ].iloc[
            :, 15:24
        ].dropna(axis='rows')
    ),
    figsize=(6,10),
    cmap='coolwarm', vmin=-2, vmax=2
)
plt.suptitle(f'{PNUM} Modified Cys Peptide Log2 Relative Ints',fontsize=14)


# We could compare it to the peptides without cysteins to get a sense of a baseline.

# In[47]:


sns.clustermap(
    np.log2(
        df5[
            df5['Cys_Peptide'] == 'No'
        ].iloc[
            :, 15:24
        ].dropna(axis='rows')
    ),
    figsize=(6,10),
    cmap='coolwarm', vmin=-2, vmax=2
)
plt.suptitle(f'{PNUM} Peptide without Cys Log2 Relative Ints',fontsize=14)


# <h3>File Proteins</h3>

# In[67]:


df6 = pd.read_csv(
    os.getcwd() + [ n for n in proteomicTxtFiles if 'Proteins' in n ][0],
    sep='\t'
)
df6 = rename_ratios4(df6)
df6['Gene'] = [ get_gene_name(x) for x in df6['Description'] ]
print(df6.shape)
print(df6.columns)
df6.head(3)


# In[68]:


df6['Master'].unique()


# In[69]:


df6['Protein FDR Confidence Combined'].unique()


# I marked the common proteomic contaminants with the prefix "cont_" in the Uniprot accession while constructing the sequence database.<br>
# Now we could look at the number and relative intensitiy profile of the contaminants.

# In[70]:


df6cont = df6[ df6['Accession'].str.contains('cont_') ].copy()
df6cont = df6cont[
    ( df6cont['Master'] == 'IsMasterProtein' )
]
df6cont.shape


# In[71]:


df6cont.set_index('Gene').iloc[:, 17:26].head(3)


# In[72]:


f = plt.figure(figsize=(6,4))
sns.heatmap(
    np.log2(
        df6cont.set_index('Gene').iloc[:, 17:26].T
    ),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=2, linecolor='white'
)
plt.suptitle(f'{PNUM} Log2 Relative Intensities of Proteomic Contaminants',fontsize=16)


# Excessive amounts of contamination in some of the samples might influence the results negatively,<br>
# especially when samples have low abundance.

# Let's proceed with non-contaminant proteins

# In[73]:


df6 = df6[ ~df6['Accession'].str.contains('cont_') ]
df6 = df6[
    ( df6['Master'] == 'IsMasterProtein' )
]
df6.set_index('Accession', inplace=True)
print(df6.shape)
df6.head(3)


# In[74]:


print( [ (i, c) for i, c in enumerate(df6.columns) ] )


# In[75]:


np.log2( df6.iloc[:, 16:25].dropna(axis='rows') ).head(3)


# In[76]:


pca_coords, loadings = pca_on_columns(
    np.log2(
        df6.iloc[:, 16:25].dropna(axis='rows')
    ),
    f'{PNUM} PCA on Proteins', figWidth = 8
)


# In[77]:


pca_coords, loadings = pca_on_columns(
    np.log2(
        df6.iloc[:, 16:25].dropna(axis='rows')
    ),
    f'{PNUM} PCA on Proteins', figWidth = 8,
    compToPlot = (2,3)
)


# In[ ]:





# In[78]:


f = plt.figure(figsize=(7,6))
sns.heatmap(
    np.log2(
        df6.iloc[:, 16:25].dropna(axis='rows')
    ).corr(method='pearson').round(2),
    square=True, cmap='coolwarm', vmin=-1, vmax=1,
    annot=False, cbar=True, linewidth=2, linecolor='white'
)
plt.suptitle(f'{PNUM} Pearson Correlations on Protein Level',fontsize=18)


# In[79]:


sns.pairplot(
    np.log2(
        df6.iloc[:, 16:25].dropna(axis='rows')
    ),
    vars=df6.iloc[:, 16:25].columns,
    diag_kind='kde', kind='scatter', markers=',', height = 1.5,
    plot_kws={'s': 8,  'alpha':0.7, 'color': 'navy'}
)
plt.suptitle(f'{PNUM} Pair Plots on Protein Level', fontsize=18)


# In[ ]:




