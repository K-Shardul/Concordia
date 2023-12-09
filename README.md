# FEniCs Workflow

FEniCs Installation: \\
Part1 - https://youtu.be/vG_2A3bKmnw
Part2 - https://youtu.be/XmsTL_d9BtQ
GMSH: 
Installation - https://gmsh.info/

Run the .geo file by clicking on “File -> Open”
Click on the “Mesh” drop down and select “3D” OR “Optimize 3D”
Save the mesh by clicking on “File -> Save Mesh”. .msh file would be saved in the predefined path of the PC.
Change the location of this .msh file to the home directory, where FEniCs is installed and all other Fenics’ Python files reside. To carry out this transfer in Ubuntu follow the below commands:
cd /mnt/f/Gmsh/Gmsh\ projects/tutorials/ __(move to the .msh file’s location. In this case it was F:/Gmsh/Gmsh projects/tutorials)
cp Cantilever_taper_beam.msh ~ __(copy .msh file to home directory)
cd ~ __(change directory to home directory)
ls __(to verify the .msh file is present in the home directory)
If need be: rm Cantilever_taper_beam.msh __(delete the already existing file)
The file names used are for example purposes. You might have different file names. Keep that in mind.

FEniCs Scripts:
Do not forget to install meshio (pip install meshio / conda install meshio)
https://drive.google.com/drive/folders/1LGQfBS9uHBreDUSDm84YIFJMOwh3Hz7r?usp=drive_link
Run the python scripts in this drive folder on JupyterNotebook. 

xyz
