#!/usr/bin/env python3

# Author: Steven Palma Morera

# Modules
import numpy as np
import matplotlib.pyplot as plt

# Main class
class occupancy_mapping_algorithm:

    # Initialize variables
    def __init__ (self,pof,poo,pinit,meas,meas_limit,map_res,map_length):
        self.pof= pof # Prob of occupancy of a cell before a measurement
        self.poo= poo # Prob of occupancy of a cell after a measurement but less than measurement+meas_limit
        self.pinit= pinit # Initial prob of all cells, how much we know from our map
        self.meas= meas # Set of measurments
        self.meas_limit=meas_limit # End of perceptual field of the sensor
        self.map_res= map_res # Length of each cell
        self.map_length= map_length+1 # Map length
    
    # Computes new the log inverse sensor model for a given cell and measurement
    # returns l(mi|zj,xj)
    def log_inv_sensor_model (self,i,j):
        # If cell before measurement, then assign the log prob of occupancy of a cell before the measurement
        if self.cells[j]<self.meas[i]: 
            return np.log(self.pof/(1-self.pof))
        # Otherwise, assign the log prob of occupancy of a cell after the measurement
        else:
            return np.log(self.poo/(1-self.poo))
    
    # Iterates the l(mi|z1:j-1,x1:j-1)
    def occupancy_grid_mapping(self,l0,logodds):

        for i in range(len(self.meas)):
            for j in range(len(self.cells)):
                # If out of range, dont update
                if self.cells[j]>self.meas[i]+ self.meas_limit:
                    logodds[j]=logodds[j]
                # Otherwise, update the logodds with the recursive, inverse sensor model term and prior term
                else:
                    logodds[j]=logodds[j]-l0[j] + self.log_inv_sensor_model(i,j)

                    # Saturating the value of logodds
                    if (logodds[j]>5):
                        logodds[j]=5
                    elif (logodds[j]<-5):
                        logodds[j]=-5

                
            # Comes back to probability
            m= 1 - 1./(1+np.exp(logodds))
            self.update_imgmap(m,i)

        # Converting matrix to image
        plt.imshow(self.map,cmap='gray')
        plt.xlabel("Distance [mm]")
        plt.ylabel('Number of measurement times ' + str(10*self.map_res))
        plt.savefig('map.png')
        plt.close()

        # Plotting the results
        plt.step(self.cells,m)
        plt.xlim(0, self.map_length)
        plt.ylim(0, 1.05)
        plt.xlabel("Distance [cm]") 
        plt.ylabel("Final Occupancy probability [Prob(mi|z1:t,x1:t)]") 
        plt.savefig("Occupancy_probability_graph.png")

        return m # Returns l(mi|z1:j,x1:j)

    # Assign the m value to all the matrix elements of a cell
    def update_imgmap(self,m,k):
        
        for i in range(len(self.cells)-1): # ignoring the value for 0
            self.map[(10*k*self.map_res):(10*self.map_res*(k+1)),(10*i*self.map_res):(10*(i+1)*self.map_res)]= np.ones([10*self.map_res,10*self.map_res])* (1-m[i+1]) #assigning intensity equal to prob of being free
            self.map[(10*self.map_res*(k+1))-1,(10*i*self.map_res):(10*(i+1)*self.map_res)]= 0.5 # to visualize where the cell ends
            self.map[(10*k*self.map_res):(10*self.map_res*(k+1)),(10*(i+1)*self.map_res)-1]= 0.5 # to visualize where the cell ends

    
    # Calls the algorithm 
    def main_mapping(self):

        # Get the grid
        self.cells= range(0,self.map_length,self.map_res)
        
        # Creates a map matrix
        self.map = np.zeros((10*self.map_res*(len(self.meas)), 10*self.map_res*(len(self.cells)-1)))
        
        # Computes the prior term based of the initial prob of all cells
        # Important: This should be log=(1-pinit/pini) because its definition its inverted in the equation
        l0= np.ones(len(self.cells))*np.log((1-self.pinit)/self.pinit)
        
        # The initial logodd value should also use the initial prob of all cells!
        logodds= np.ones(len(self.cells))*np.log(self.pinit/(1-self.pinit))
        
        # Update the logoods for the entire map
        m= self.occupancy_grid_mapping(l0,logodds)

        print("Prob(mi|z1:t,x1:t):",m)

        return 0

# Main 
def main():

    # Customize values of the map and sensor
    pof=0.2 # Prob of occupancy of a cell before a measurement
    poo=0.7 # Prob of occupancy of a cell after a measurement but less than measurement+meas_limit
    pinit=0.5  # Initial prob of all cells, how much we know from our map
        
    meas=np.array([19,21,26,23,24,79,73,75,76,77,]) # # Set of measurements
    meas_limit=20 # End of perceptual field of the sensor
    map_res=10 # Length of each cell
    map_length=100 # Line length

    # Create an instance of the class
    map = occupancy_mapping_algorithm(pof,poo,pinit,meas,meas_limit,map_res,map_length)
    map.main_mapping() # Run the algorithm

    return 0

if __name__ == '__main__':
    main()