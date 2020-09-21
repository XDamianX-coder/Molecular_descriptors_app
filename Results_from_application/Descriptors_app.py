######################
# Import libraries
######################
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw
from PIL import Image
import time
import matplotlib.pyplot as plt
import pubchempy as pcp

######################
# Custom function
######################
## Calculate molecular descriptors
from typing import Any


def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  try:
      AromaticAtom = sum(aa_count)
      HeavyAtom = Descriptors.HeavyAtomCount(m)
      AR = AromaticAtom/HeavyAtom
  except:
      pass
  return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)
        desc_NumRadicalElectrons = Descriptors.NumRadicalElectrons(mol)
        desc_num_valence_Electrons = Descriptors.NumValenceElectrons(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion,
                        desc_NumRadicalElectrons,
                        desc_num_valence_Electrons])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion","NumRadicalElectrons","NumValenceElectrons"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

def generate_2(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MAXAbs = Descriptors.MaxAbsPartialCharge(mol)
        desc_MAXPartial = Descriptors.MaxPartialCharge(mol)
        desc_MINABS = Descriptors.MinAbsPartialCharge(mol)
        desc_MINPartial = Descriptors.MinPartialCharge(mol)
        desc_FPDEN1 = Descriptors.FpDensityMorgan1(mol)
        desc_FPDEN3 = Descriptors.FpDensityMorgan3(mol)

        row = np.array([desc_MAXAbs,
                        desc_MAXPartial,
                        desc_MINABS,
                        desc_MINPartial,
                        desc_FPDEN1,
                        desc_FPDEN3])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MaxAbsPartialCharge","MaxPartialCharge","MinAbsPartialCharge","MinPartialCharge","FpDensityMorgan1","FpDensityMorgan3"]
    descriptors_1 = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors_1

def cid_to_smiles(smiles):
    baseData_1=[]
    p= pcp.get_compounds(smiles, 'smiles')
    baseData_1.append(p)
    baseData_1 = np.arange(1, 1)
    i = 0
    row = np.array([p])
    if (i == 0):
        baseData_1 = row
    else:
        baseData_1 = np.vstack([baseData_1, row])
    i = i + 1
    columnNames = ["Compound CID"]
    cid = pd.DataFrame(data=baseData_1, columns=columnNames)
    return cid

def molecular_formula(number):
    baseData_2 = []
    g = pcp.Compound.from_cid(number)
    #print(g.molecular_formula)
    baseData_2.append(g.molecular_formula)
    baseData_2 = np.arange(1, 1)
    i = 0
    row = np.array([g.molecular_formula])
    if (i == 0):
        baseData_2 = row
    else:
        baseData_2 = np.vstack([baseData_2, row])
    i = i + 1
    columnNames = ["Molecular Formula"]
    MF = pd.DataFrame(data=baseData_2, columns=columnNames)
    return MF

def iupac_name(number):
    baseData_3 = []
    u = pcp.Compound.from_cid(number)
    baseData_3.append(u.iupac_name)
    baseData_3 = np.arange(1, 1)
    i = 0
    row = np.array([u.iupac_name])
    if (i == 0):
        baseData_3 = row
    else:
        baseData_3 = np.vstack([baseData_3, row])
    i = i + 1
    columnNames = ["IUPAC name"]
    IUPAC = pd.DataFrame(data=baseData_3, columns=columnNames)

    return IUPAC


def search_by_cpnd_name(name):

    result = pcp.get_compounds(name,'name')
    baseData_4 = []
    baseData_4.append(result)
    baseData_4 = np.arange(1, 1)
    i = 0
    row = np.array([result])
    if (i == 0):
        baseData_4 = row
    else:
        baseData_4 = np.vstack([baseData_4, row])
    i = i + 1
    columnNames = ["List of compounds"]
    List_cpnds = pd.DataFrame(data=baseData_4, columns=columnNames)

    return List_cpnds


def IMAGE(smiles, basename='compound'):
    molData=[]
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        molData.append(mol)
    filenames = []
    for i, mol in enumerate(molData):
        filename = "%s%d.png" % (basename, i)
        Draw.MolToFile(mol, filename)
        filenames.append(filename)
    return filenames



######################
# Page Title
######################

image = Image.open('chem1.png')

st.image(image, use_column_width=True)

st.write("""
# Molecular Descriptors Web App
This app calculate the **Molecular Descriptors** of compounds!\n
Author: **Damian Nowak**
""")

my_bar = st.progress(0)
for percent_complete in range(100):
    time.sleep(0.03)
    my_bar.progress(percent_complete + 1)


######################
# Input molecules (Side Panel)
######################

st.sidebar.header('User Input Features')

## Read SMILES input
SMILES_input = "O=C1CN=C(C2=CC=CC=C2)C2=C(N1)C=CC=C2\nCNC(C)CC1=CC=CC=C1\nCCN(CC)C(=O)C1CN(C)C2CC3=CNC4=CC=CC(=C34)C2=C1\nCOC(=O)C1C2CCC(CC1OC(=O)C1=CC=CC=C1)N2C"

SMILES = st.sidebar.text_area("SMILES input.", SMILES_input)
SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

st.header('Input SMILES.')
SMILES[1:] # Skips the dummy first item
time.sleep(0.1)
## Calculate molecular descriptors
st.header('Computed molecular descriptors.')
X = generate(SMILES)
X[1:] # Skips the dummy first item
Y = generate_2(SMILES)
Y[1:]
Z = IMAGE(SMILES)
time.sleep(0.2)
basename = 'compound'
molData=[]
for i in SMILES:
    mol = Chem.MolFromSmiles(i)
    molData.append(mol)
filenames =[]
for i, mol in enumerate(molData):
    filename = "%s%d.png" % (basename, i)
    filenames.append(filename)

name = filenames[1:]
st.header('Images of the molecules.')
#Y = openimg(name)
#Y # Skips the dummy first item (not now)
st.image(name)
time.sleep(0.1)
new = generate(SMILES)
fig, ax = plt.subplots()
ax.hist(new['MolWt'], bins=50)
st.header('Distribution of molecular weight.')
st.pyplot(fig)



time.sleep(0.1)
new = generate(SMILES)
fig, ax = plt.subplots()
ax.hist(new['AromaticProportion'], bins=50)
st.header('Distribution of aromatic proportion.')
st.pyplot(fig)

time.sleep(0.1)
new = generate(SMILES)
fig, ax = plt.subplots()
ax.hist(new['NumValenceElectrons'], bins=50)
st.header('Distribution of the number of valence electrons.')
st.pyplot(fig)

time.sleep(0.1)
new = generate_2(SMILES)
fig, ax = plt.subplots()
ax.hist(new["MaxPartialCharge"], bins=50)
st.header('Distribution of the maximal partial charge.')
st.pyplot(fig)

time.sleep(0.1)
new = generate_2(SMILES)
fig, ax = plt.subplots()
ax.hist(new["MinPartialCharge"], bins=50)
st.header('Distribution of the minimal partial charge.')
st.pyplot(fig)


SMILES_input_1 = "O=C1CN=C(C2=CC=CC=C2)C2=C(N1)C=CC=C2"
SMILES_1 = st.sidebar.text_area("SMILES input to get CID", SMILES_input_1)
SMILES_1 = SMILES_1.split('\n')

st.header('Input SMILES to get the compound cid.')
SMILES_1
A = cid_to_smiles(SMILES_1)
A





CID_input = "258"
CID = st.sidebar.text_area("CID input to get molecular formula.", CID_input)
CID = CID.split('\n')

st.header('Input CID to get molecular formula.')
CID
B = molecular_formula(CID)
B

CID_input_1 = "11786"
CID_1 = st.sidebar.text_area("CID input to get IUPAC name.", CID_input_1)
CID_1 = CID_1.split('\n')

st.header('Input CID to get IUPAC name of compound.')
CID_1
C = iupac_name(CID_1)
C

Name_input = "Glucose"
Name_1 = st.sidebar.text_area("Name input to CID number.", Name_input)
Name_1 = Name_1.split('\n')

st.header('Input name to CID number.')
Name_1
D = search_by_cpnd_name(Name_1)
D

