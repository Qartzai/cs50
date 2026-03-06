from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import logging

# Suppress pgmpy info/warning messages
logging.getLogger('pgmpy').setLevel(logging.ERROR)
# Define structure
model = BayesianNetwork([
    ('rain', 'maintenance'),
    ('rain', 'train'),
    ('maintenance', 'train'),
    ('train', 'appointment')
])

# Rain CPD
cpd_rain = TabularCPD('rain', 3, [[0.7], [0.2], [0.1]],
                      state_names={'rain': ['none', 'light', 'heavy']})

# Maintenance | Rain
cpd_maint = TabularCPD('maintenance', 2, 
                       [[0.4, 0.2, 0.1],
                        [0.6, 0.8, 0.9]],
                       evidence=['rain'], evidence_card=[3],
                       state_names={'maintenance': ['yes', 'no'],
                                   'rain': ['none', 'light', 'heavy']})

# Train | Rain, Maintenance
cpd_train = TabularCPD('train', 2,
                       [[0.8, 0.9, 0.6, 0.7, 0.4, 0.5],
                        [0.2, 0.1, 0.4, 0.3, 0.6, 0.5]],
                       evidence=['rain', 'maintenance'], evidence_card=[3, 2],
                       state_names={'train': ['on time', 'delayed'],
                                   'rain': ['none', 'light', 'heavy'],
                                   'maintenance': ['yes', 'no']})

# Appointment | Train  
cpd_appt = TabularCPD('appointment', 2,
                      [[0.9, 0.6],
                       [0.1, 0.4]],
                      evidence=['train'], evidence_card=[2],
                      state_names={'appointment': ['attend', 'miss'],
                                  'train': ['on time', 'delayed']})

model.add_cpds(cpd_rain, cpd_maint, cpd_train, cpd_appt)
model.check_model()

# Query
from pgmpy.inference import VariableElimination
infer = VariableElimination(model)
result = infer.query(['appointment'], evidence={'rain': 'heavy', 'maintenance': 'no', 'train': 'delayed'})
print(result)