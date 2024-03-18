def division1(input_data):
    # Define persistent variable
    if not hasattr(division1, 'temp_input'):
        division1.temp_input = input_data + 1
    
    # Perform the division operation
    output = input_data / division1.temp_input
    
    # Check for NaN and handle it
    if np.isnan(output):
        output = 1
    
    # Update the persistent variable
    division1.temp_input = input_data
    
    return output