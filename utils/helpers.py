def get_bound(neuron_number, 
              distance, 
              total_width,):
    """
    Calculate Boundary for Given Total Width, Neuron_number and Distance
    """
    bound = (total_width - distance * (neuron_number - 1)) / 2
    return bound
