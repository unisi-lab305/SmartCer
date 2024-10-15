# smart-rec
Smart Renewable Energy Community (REC) simulator 

---

## Key Performance Indicators (KPIs) 

### 1. Net Energy Expense

Formula:
Net Energy Expense = Σ(energy_expense) - Σ(energy_revenue)

Meaning:
This KPI represents the total cost incurred for energy consumption after accounting for any revenues from energy sales. 
A positive value indicates higher expenses than revenues.

---

### 2. Tip Potential

Formula:
Tip Potential = Σ(tip_potential)

Meaning:
This indicates the total potential energy that can be optimized or "tipped" towards more efficient usage or generation. 
It reflects the opportunities available for improving energy performance.

---

### 3. Tip Actual

Formula:
Tip Actual = Σ(tip_actual)

Meaning:
This measures the actual amount of energy that has been optimized or "tipped." Comparing this with the Tip Potential 
helps assess how effectively opportunities for energy savings are being utilized.

---

### 4. Self Consumption

Formula:
Self Consumption = (Σ(selfconsumption) / Σ(exported_energy)) × 100

Meaning:
This percentage reflects how much energy generated is consumed on-site compared to what is exported. A higher percentage 
indicates effective utilization of self-generated energy.

---

### 5. Energy Demand

Formula:
Energy Demand = Σ(imported_energy)

Meaning:
This KPI indicates the total energy imported from external sources. It helps assess reliance on outside energy supplies.

---

### 6. Gross Self Consumption Rate

Formula:
Gross Self Consumption Rate = (Σ(selfconsumption) / (Σ(imported_energy) + Σ(selfconsumption))) × 100

Meaning:
This percentage shows how much of the total energy consumed comes from self-generated energy, including both imported 
and self-consumed energy. It provides insight into the sustainability of energy use.

---

### 7. Net Metering

Formula:
Net Metering = Σ(selfconsumption) - Σ(exported_energy)

Meaning:
This KPI indicates the difference between the energy consumed and the energy exported. A positive value shows a net gain
in self-consumption.

---

### 8. Cost Per Unit Energy

Formula:
Cost Per Unit Energy = Net Energy Expense / Energy Demand

Meaning:
This indicates the average cost incurred per unit of energy consumed. It helps evaluate the economic efficiency of 
energy use.

---

### 9. Revenue Per Unit Energy

Formula:
Revenue Per Unit Energy = Σ(energy_revenue) / Σ(exported_energy)

Meaning:
This KPI measures the average revenue earned for each unit of energy exported. It provides insight into the 
profitability of energy sales.

---

## #10. Energy Balance

Formula:
Energy Balance = Σ(imported_energy) - (Σ(selfconsumption) + Σ(exported_energy))

Meaning:
This KPI indicates the overall balance between imported energy and the sum of energy consumed and exported. A positive 
value suggests a surplus, while a negative value indicates a deficit.