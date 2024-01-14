import vtk
from vtk import vtkOpenGLPolyDataMapper, vtkActor
import numpy as np
from skimage import io
import os
import csv
import math
from tkinter import filedialog
import tkinter as tk
import multiprocessing as mp
import threading
from vtkmodules.util.numpy_support import vtk_to_numpy
from matplotlib import cm
import time

def run_vtk_analysis(input_directory, file_name, output_file_path):
    if file_name.lower().endswith('.tif'):
        file_path = os.path.join(input_directory, file_name)
        probability_volume = io.imread(file_path)
        probability_volume = probability_volume.astype(np.uint8)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(probability_volume.shape[2], probability_volume.shape[1], probability_volume.shape[0])
        image_data.SetSpacing(0.1, 0.1, 0.1)

        vtk_array = vtk.vtkUnsignedCharArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetArray(probability_volume.ravel(), np.prod(probability_volume.shape), False)
        image_data.GetPointData().SetScalars(vtk_array)

        # Create a Marching Cubes filter
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(image_data)
        marching_cubes.SetValue(0, 0.5)  # Adjust the threshold as needed
        
        fill_holes = vtk.vtkFillHolesFilter()
        fill_holes.SetInputConnection(marching_cubes.GetOutputPort())
        fill_holes.SetHoleSize(1000)  # Adjust hole size as needed

        # Create a connectivity filter
        connectivity_filter = vtk.vtkConnectivityFilter()
        connectivity_filter.SetInputConnection(fill_holes.GetOutputPort())
        connectivity_filter.SetExtractionModeToLargestRegion()  # Keep only the largest connected region

        # Create a mapper for the mesh
        mesh_mapper = vtkOpenGLPolyDataMapper()
        mesh_mapper.SetInputConnection(connectivity_filter.GetOutputPort())
        
        mass_properties = vtk.vtkMassProperties()
        mass_properties.SetInputConnection(connectivity_filter.GetOutputPort())
        mass_properties.Update()
        
        surface_area = round(mass_properties.GetSurfaceArea(), 3)
        volume = round(mass_properties.GetVolume(), 3)
        sphericity = round(((math.pi**(1/3))*((6*volume)**(2/3))/surface_area), 3)
        
        print(f"Writing to CSV: {file_name, surface_area, volume}")
        with open(output_file_path, 'a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([file_name, surface_area, volume, sphericity])
        
class InputAndOutputManager:
    def __init__(self):
        # Create the main Tkinter window
        root = tk.Tk()
        root.grid()
        self.root = root
        self.root.title("Input and Output Manager")
        self.root.grid()
        self.data = []
        self.currentindex = 0
        
        self.meshrenderer = None
        self.render_window = None
        self.render_window_interactor = None
        self.renderer = None
        
        self.meshactor = None
        self.text_actor = None
        self.marching_cubes = None
        self.fill_holes = None
        self.connectivity_filter = None
        self.mesh_mapper = None
        
        self.input_directory = tk.StringVar()
        self.output_file = tk.StringVar()
        self.file_list = None

        # Create labels and entry widgets
        tk.Label(root, text="Input Directory:").grid(row=0, column=0)
        tk.Entry(root, textvariable=self.input_directory, state='readonly').grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_input).grid(row=0, column=2)

        tk.Label(root, text="Output File:").grid(row=1, column=0)
        tk.Entry(root, textvariable=self.output_file, state='readonly').grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Create an "Analyze" button
        tk.Button(root, text="Analyze", command=self.analyze).grid(row=2, column=1)
        
        # Create a "Visualise" button
        tk.Button(root, text="Visualise", command=self.transfer).grid(row=3, column=1)
        
        self.root.mainloop()
    
    def add_text_annotation(self):
        if hasattr(self, 'text_actor'):
            self.renderer.RemoveActor2D(self.text_actor)
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetTextScaleModeToNone()
        self.text_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedDisplay()
        initial_position = (0.01, 3)
         
        window_size = self.render_window.GetSize()
        font_size = min(window_size) // 25  # Adjust the factor as needed
        self.text_actor.GetTextProperty().SetFontSize(font_size)
        self.text_actor.GetTextProperty().SetColor(0.1, 0.1, 0.1)  # White text
        self.text_actor.SetInput(self.text)
        self.renderer.AddActor2D(self.text_actor)
        self.render_window.Render()
        
    def on_window_resize(self, obj, event):
        # Handle window resize event
        window_size = self.render_window.GetSize()
        self.renderer.RemoveActor2D(self.text_actor)
        font_size = min(window_size) // 25  # Adjust the factor as needed
        self.text_actor.GetTextProperty().SetFontSize(font_size)
        self.text_actor.GetTextProperty().SetColor(0.1, 0.1, 0.1)  # White text
        self.text_actor.SetInput(self.text)
        self.renderer.AddActor2D(self.text_actor)
        self.render_window.Render()

    def browse_input(self):
        input_dir = filedialog.askdirectory()
        if input_dir:
            self.input_directory.set(input_dir)

    def browse_output(self):
        output_file = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if output_file:
            self.output_file.set(output_file)

    def write_csv_header(self, file_path):
        with open(file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["File", "Surface Area", "Volume", "Sphericity"])
            
    def read_csv(self):
        data = []
        with open(self.output_file.get(), mode ='r') as csvfile:
            csv_reader = csv.reader(csvfile)
            next(csv_reader)
            for row in csv_reader:
                data.append(row)
        self.data = data

    def analyze(self):
        input_directory = self.input_directory.get()
        if len(input_directory) > 0:
            self.file_list = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
            output_file_path = self.output_file.get()
            self.write_csv_header(output_file_path)
            for i in self.file_list:    
                p1 = mp.Process(target = run_vtk_analysis, args =  (input_directory, i, output_file_path))
                p1.start()
            p1.join()
    
    def calculate_centroid(self, data):
        num_points = len(data)
        i = 0
        x = 0
        y = 0
        z = 0
        while i < num_points:
            x += data[i][0]
            y += data[i][1]
            z += data[i][2]
            i += 1
        x = x/num_points
        y = y/num_points
        z = z/num_points
        centroid = [x,y,z]
        return centroid
            
    def load_next_mesh(self, current_data):
        # Load probabilities from an image
        probability_volume = io.imread(self.input_directory.get() + '/' + str(current_data[0]))
        probability_volume = probability_volume.astype(np.uint8)

        # Create a VTK volume
        image_data = vtk.vtkImageData()
        image_data.SetDimensions(probability_volume.shape[2], probability_volume.shape[1], probability_volume.shape[0])
        image_data.SetSpacing(0.1, 0.1, 0.1)  # Adjust as needed

        # Convert numpy array to VTK array
        vtk_array = vtk.vtkUnsignedCharArray()
        vtk_array.SetNumberOfComponents(1)
        vtk_array.SetArray(probability_volume.ravel(), np.prod(probability_volume.shape), False)
        image_data.GetPointData().SetScalars(vtk_array)  

        # Create a Marching Cubes filter
        self.marching_cubes = vtk.vtkMarchingCubes()
        self.marching_cubes.SetInputData(image_data)
        self.marching_cubes.SetValue(0, 1)  # Adjust the threshold as needed
        #self.remove_points_by_distance()
        
        self.fill_holes = vtk.vtkFillHolesFilter()
        self.fill_holes.SetInputConnection(self.marching_cubes.GetOutputPort())
        self.fill_holes.SetHoleSize(1000)  # Adjust hole size as needed
        
        # Create a connectivity filter
        self.connectivity_filter = vtk.vtkConnectivityFilter()
        self.connectivity_filter.SetInputConnection(self.fill_holes.GetOutputPort())
        self.connectivity_filter.SetExtractionModeToLargestRegion()  # Keep only the largest connected region
        
        self.normals = vtk.vtkPolyDataNormals()
        self.normals.SetInputConnection(self.connectivity_filter.GetOutputPort())
        self.normals.ComputePointNormalsOn()
        self.normals.ComputeCellNormalsOff()
        self.normals.SplittingOff()
        self.normals.Update()
        
        # Assuming 'points' is a vtkPoints object containing the mesh points
        mesh_points = self.normals.GetOutput().GetPoints()
        
        # Extract point coordinates as a NumPy array
        points_array = vtk_to_numpy(mesh_points.GetData())
        centroid = self.calculate_centroid(points_array)

        # Reshape points_array to have shape (95274, 3)
        points_array = points_array.reshape(-1, 3)

        # Compute distances from each point to the centroid
        distances = np.linalg.norm(points_array - centroid, axis=1)
        
        # Set up scalar values for the mapper based on distances
        scalar_values = vtk.vtkDoubleArray()
        scalar_values.SetNumberOfTuples(len(distances))
        scalar_values.SetNumberOfComponents(1)

        # Assign distances as scalar values
        for i, distance in enumerate(distances):
            scalar_values.SetValue(i, distance)

        # Create a mapper for the mesh
        self.mesh_mapper = vtkOpenGLPolyDataMapper()
        self.mesh_mapper.SetInputConnection(self.normals.GetOutputPort())
        self.mesh_mapper.SetScalarModeToUsePointData()
        self.mesh_mapper.ScalarVisibilityOn()
        self.mesh_mapper.SetScalarRange(np.min(distances), np.max(distances))
        self.mesh_mapper.GetInput().GetPointData().SetScalars(scalar_values)
        # Use a predefined colormap from matplotlib
        cmap = cm.viridis

        # Create a lookup table from the colormap
        lut = vtk.vtkLookupTable()
        lut.SetNumberOfColors(256)  # Adjust the number of colors as needed
        for i, color in enumerate(cmap(np.linspace(0, 1, 256))):
            lut.SetTableValue(i, color[0], color[1], color[2], 1.0)
            
        self.mesh_mapper.SetLookupTable(lut)
        
        # Create an actor for the mesh
        self.mesh_actor = vtkActor()
        
        self.mesh_actor.SetMapper(self.mesh_mapper)
        self.mesh_actor.GetMapper().SetLookupTable(lut)
        self.mesh_actor.GetMapper().ScalarVisibilityOn()

        # Add the new mesh actor to the existing renderer
        self.renderer.AddActor(self.mesh_actor)

        # Request a render to update the scene
        self.render_window.Render()
        
        # Display metadata as text annotation
        self.text = f"File: {current_data[0]}\nSurface Area: {current_data[1]}\nVolume: {current_data[2]}\nSphericity: {current_data[3]}"
        self.add_text_annotation()
        self.render_window.AddObserver("ModifiedEvent", self.on_window_resize)
        
    def transfer(self):
        def CreateButtonOff(image):
            white = [255, 255, 255]
            CreateImage(image, white, white)

        def CreateButtonOn(image):
            white = [255, 255, 255]
            CreateImage(image, white, white)

        def CreateImage(image, color1, color2):
            size = 12
            dims = [size, size, 1]
            lim = size / 3.0

            # Specify the size of the image data
            image.SetDimensions(dims[0], dims[1], dims[2])
            arr = vtk.vtkUnsignedCharArray()
            arr.SetNumberOfComponents(3)
            arr.SetNumberOfTuples(dims[0] * dims[1])
            arr.SetName('scalars')

            # Fill the image with
            for y in range(dims[1]):
                for x in range(dims[0]):
                    if x >= lim and x < 2 * lim and y >= lim and y < 2 * lim:
                        arr.SetTuple3(y*size + x, color2[0], color2[1], color2[2])
                    else:
                        arr.SetTuple3(y*size + x, color1[0], color1[1], color1[2])

            image.GetPointData().AddArray(arr)
            image.GetPointData().SetActiveScalars('scalars')
                
        #############################
        self.read_csv()
        current_data = [self.data[0][0], self.data[0][1], self.data[0][2], self.data[0][3]]

        # Create a renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.9, 0.9, 0.9)
    
        # Create a render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)

        # Create a render window interactor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        # Set up interactor style for the mesh
        mesh_interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.render_window_interactor.SetInteractorStyle(mesh_interactor_style)
        self.root.destroy()

        def rotate_camera_callback():
            self.renderer.GetActiveCamera().Azimuth(0.5)  # Adjust the rotation speed as needed
            self.render_window.Render()

        self.render_window_interactor.AddObserver('TimerEvent', lambda obj, event: rotate_camera_callback())
        self.render_window_interactor.CreateRepeatingTimer(10)  # Adjust the timer interval as needed
        
        self.load_next_mesh(current_data)
        
        ############################################
        # Create two images for texture
        image1 = vtk.vtkImageData()
        image2 = vtk.vtkImageData()
        CreateButtonOff(image1)
        CreateButtonOn(image2)
        
        # Create the widget and its representation
        buttonRepresentation = vtk.vtkTexturedButtonRepresentation2D()
        buttonRepresentation.SetNumberOfStates(2)
        buttonRepresentation.SetButtonTexture(0, image1)
        buttonRepresentation.SetButtonTexture(1, image2)

        buttonWidget = vtk.vtkButtonWidget()
        buttonWidget.SetInteractor(self.render_window_interactor)
        buttonWidget.SetRepresentation(buttonRepresentation)
        
        def try_callback(func, *args):
            """Wrap a given callback in a try statement."""
            import logging
            try:
                func(*args)
            except Exception as e:
                logging.warning('Encountered issue in callback: {}'.format(e))
            return

        def callback(val):
            print("Button pressed!")
            input_directory = self.input_directory.get()
            # Check if the contents of the input directory have changed
            current_files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]
            if current_files != self.file_list:
                self.mesh_actor.SetVisibility(False)
                self.file_list = current_files
                if len(input_directory) > 0:
                    output_file_path = self.output_file.get()
                    with open(output_file_path, 'w', newline='') as csvfile:
                        csv_writer = csv.writer(csvfile)
                        csv_writer.writerow(["File", "Surface Area", "Volume", "Sphericity"])
                    for i in self.file_list:    
                        p1 = mp.Process(target = run_vtk_analysis, args =  (input_directory, i, output_file_path))
                        p1.start()
                p1.join()
                time.sleep(1)           
                self.read_csv()
    
            if (self.currentindex + 1) < len(self.data):
                self.currentindex += 1
            else:
                self.currentindex = 0

            current_data = [self.data[self.currentindex][0], self.data[self.currentindex][1], self.data[self.currentindex][2], self.data[self.currentindex][3]]

            if hasattr(self, 'mesh_actor'):
                self.mesh_actor.SetVisibility(False)
                self.render_window.Render()

            self.load_next_mesh(current_data)

        def _the_callback(widget, event):
            value = widget.GetRepresentation().GetState()
            if hasattr(callback, '__call__'):
                try_callback(callback, value)
            return
            
        buttonWidget.AddObserver(vtk.vtkCommand.StateChangedEvent, _the_callback)
        
        # Place the widget. Must be done after a render so that the
        # viewport is defined.
        # Here the widget placement is in normalized display coordinates
        upperRight = vtk.vtkCoordinate()
        upperRight.SetCoordinateSystemToNormalizedDisplay()
        upperRight.SetValue(100.0, 100.0)
        
        bds = [0]*6
        sz = 50.0
        bds[0] = upperRight.GetComputedDisplayValue(self.renderer)[0] - sz
        bds[1] = bds[0] + sz
        bds[2] = upperRight.GetComputedDisplayValue(self.renderer)[1] - sz
        bds[3] = bds[2] + sz
        bds[4] = bds[5] = 0.0

        # Scale to 1, default is .5
        buttonRepresentation.SetPlaceFactor(1)
        buttonRepresentation.PlaceWidget(bds)
        buttonWidget.On()
          
        self.render_window_interactor.Start()
    
if __name__ == '__main__':
# Create an instance of the ImageAnalysisGUI class
    gui = InputAndOutputManager()

    
