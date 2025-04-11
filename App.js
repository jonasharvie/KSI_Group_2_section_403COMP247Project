import React, { useState, useEffect } from "react";
import axios from "axios";
import Select from "react-select";
import "./App.css"; 

function App() {
  const [formData, setFormData] = useState({});
  const [dataset, setDataset] = useState([]);
  const [prediction, setPrediction] = useState("");
  const [error, setError] = useState("");

  // Load dataset on component mount
  useEffect(() => {
    const fetchOptions = async () => {
      try {
        const response = await axios.get("http://127.0.0.1:12345/get-options");
        setDataset(response.data);
      } catch (err) {
        console.error("Error fetching dropdown options:", err);
      }
    };

    fetchOptions();
  }, []);

  // dropdown fields
  const getOptions = (key) => {
    if (!dataset[key]) return [];
    return dataset[key].map((value) => ({
      label: value,
      value: value,
    }));
  };

  // Handle form changes
  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  // Handle dropdown changes
  const handleDropdownChange = (selectedOption, field) => {
    setFormData({
      ...formData,
      [field]: selectedOption ? selectedOption.value : "",
    });
  };

  // Submit form data
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    setPrediction(""); // Reset previous predictions
  
    try {
      const response = await axios.post("http://127.0.0.1:12345/predict", [formData]);
      console.log("Backend response:", response.data);
  
      // Extract prediction and save to state
      setPrediction(response.data.prediction);
    } catch (err) {
      console.error("Error during prediction:", err);
      setError("Something went wrong while making the prediction. Please try again.");
    }
  };

  return (
    <div className="app-container">
      <h1 className="app-title">Fatality Predictor</h1>
      <form onSubmit={handleSubmit} className="form-container">

        {/* Input: Date */}
        <label>
          Date:&nbsp;&nbsp;
          <input type="date" name="DATE" onChange={handleChange} required />
        </label>

        {/* Input: Time */}
        <label>
          Time: &nbsp;&nbsp;
          <input type="number" name="TIME" placeholder="24hr format" onChange={handleChange} required />
        </label>

         {/* Input: Involved Age */}
         <label>
          Age:&nbsp;&nbsp;
          <input type="number" name="INVAGE" placeholder="Individual's" onChange={handleChange} required />
        </label>
          {/* Input: Speeding */}
          <label>
          Speeding:&nbsp;&nbsp;
          <select name="SPEEDING" onChange={handleChange} required>
            <option value="">Select</option>
            <option value="Yes">Yes</option>
            <option value="No">No</option>
          </select>
        </label>
        {/* Dropdown: Neighborhood */}
        <label>
          Neighborhood:
          <Select
            options={getOptions("NEIGHBOURHOOD_158")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "NEIGHBOURHOOD_158")}
            isSearchable
            placeholder="Select or type neighborhood"
          />
        </label>

        {/* Dropdown: Street 1 */}
        <label>
          Street 1:
          <Select
            options={getOptions("STREET1")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "STREET1")}
            isSearchable
            placeholder="Select or type street 1"
          />
        </label>

        {/* Dropdown: Street 2 */}
        <label>
          Street 2:
          <Select
            options={getOptions("STREET2")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "STREET2")}
            isSearchable
            placeholder="Select or type street 2"
          />
        </label>

        {/* Dropdown: Road Class */}
        <label>
          Road Class:
          <Select
            options={getOptions("ROAD_CLASS")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "ROAD_CLASS")}
            isSearchable
            placeholder="Select road class"
          />
        </label>

        {/* Dropdown: Traffic Control */}
        <label>
          Traffic Control:
          <Select
            options={getOptions("TRAFFCTL")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "TRAFFCTL")}
            isSearchable
            placeholder="Select traffic control"
          />
        </label>

        {/* Dropdown: Visibility */}
        <label>
          Visibility:
          <Select
            options={getOptions("VISIBILITY")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "VISIBILITY")}
            isSearchable
            placeholder="Select visibility"
          />
        </label>

        <label>
  Driver Condition:
  <Select
    options={getOptions("DRIVCOND")}
    onChange={(selectedOption) => handleDropdownChange(selectedOption, "DRIVCOND")}
    isSearchable
    placeholder="Select driver condition"
  />
</label>

        {/* Dropdown: Road Surface Condition */}
        <label>
          Road Surface Condition:
          <Select
            options={getOptions("RDSFCOND")}
            onChange={(selectedOption) => handleDropdownChange(selectedOption, "RDSFCOND")}
            isSearchable
            placeholder="Select surface condition"
          />
        </label>

        <label>
  Division:
  <Select
    options={getOptions("DIVISION")}
    onChange={(selectedOption) => handleDropdownChange(selectedOption, "DIVISION")}
    isSearchable
    placeholder="Select division"
  />
</label>

        {/* Dropdown: Impact Type */}
      <label>
        Impact Type:
        <Select
          options={getOptions("IMPACTYPE")}
          onChange={(selectedOption) => handleDropdownChange(selectedOption, "IMPACTYPE")}
          isSearchable
          placeholder="Select impact type"
        />
      </label>


        <button type="submit" className="submit-button">Predict</button>
      </form>

      {/* Display prediction or error */}
      {prediction && (
        <div className="prediction-result">
          <h2>Prediction Result:</h2>
          <p>{prediction}</p>
        </div>
      )}
      {error && (
        <div className="error-message">
          <h2>Error:</h2>
          <p>{error}</p>
        </div>
      )}
    </div>
  );
}

export default App;