# Predicting Industrial Machine Downtime

A classifier that tells whether a CNC machine is in a downtime state from its current sensor readings, built on 2500 readings from three machines over eight months.

The short version: the model reaches an F1 of 0.97 and the number holds up under three different ways of splitting the data. But it detects a failure that is already happening — it cannot see one coming. The task brief asks for proactive maintenance, and I show with a direct test that this data cannot support it.

## The result

| model | f1 | accuracy |
|---|---|---|
| dummy (always predicts the majority class) | 0.6720 | 0.5060 |
| logistic regression | 0.8703 | 0.8700 |
| single decision tree, depth 3 | 0.9476 | 0.9460 |
| **Bagging with Decision Trees** | **0.9704** | **0.9700** |

Recall is 0.9723 and precision is 0.9685 on a held-out test set of 500 readings.

The baselines are there on purpose. A high score means little until you know what an uninformed guess scores. Here a single tree small enough to draw by hand already reaches 0.95, so the ensemble buys about 2.5 points on top of it. That is a real gain, but a small one, and worth knowing before anyone builds a pipeline around it.

## Can the numbers be trusted?

My first version reported 0.98. It was wrong, and finding out why took most of the work.

**The missing values were filled using the target.** The original code filled each gap with the median or mean of its own class of `Downtime`. The filled values then carried a piece of the answer, and the model got a hint it would never have in real use. Moving all the filling inside a scikit-learn pipeline, where it is recomputed on the training part of every fold, costs about 1.5 points of F1. Those were not real.

**Readings taken close together could leak across the split.** Each row is one reading, and there are about six readings per machine per day. If readings from the same day look alike, a random split puts near-copies into training and into test. I checked, and they do not: from three readings a day onwards, the share of days with a single label sits at chance level.

**There could be drift over eight months.** The day-to-day correlation of the failure rate is -0.063, -0.130 and -0.022 for the three machines. That is zero.

Then I rebuilt the model under three splits:

| split | f1 | accuracy |
|---|---|---|
| random | 0.9704 | 0.9700 |
| by machine-day (no day is split) | 0.9804 | 0.9801 |
| by time (train on the first 80% of days) | 0.9873 | 0.9874 |

All three agree within one point, which on 500 test rows is five rows. The score is real.

## Detecting is not predicting

This is the part I did not expect to find.

The brief is not consistent with itself. The background asks for maintenance planned in advance, which is forecasting. The task then asks for a model that works with real-time data and detects a likely failure, which is a judgement about the present moment. The model above does the second thing.

So I tested whether it could do the first. I averaged the readings to one row per machine per day and asked for the same label twice — the state of that day, and the state of the next day. Same features, same models, same split by time.

| target | dummy | logistic | bagging |
|---|---|---|---|
| same day (accuracy) | 0.4878 | 0.8171 | 0.8780 |
| next day (accuracy) | 0.4756 | 0.5488 | 0.5244 |

On the next day it is a coin flip.

These readings describe the state a machine is in right now. They hold no information about the state it will be in tomorrow. That is a property of the data, not of the model: nothing in this table accumulates over time. Forecasting a failure would need hours since the last service, tool wear, a count of load cycles, the trend of a reading rather than its value — or simply readings taken far more often than a few times a day.

A model that flags a bad state as it happens is still useful. It just must not be sold as early warning.

## One model or three?

The brief has three machines on the shop floor, so a natural question is whether each should get its own model.

| machine | rows to train on | per-machine f1 | all-machines f1 |
|---|---|---|---|
| Makino-L1-Unit1-2013 | 695 | 0.9780 | 0.9724 |
| Makino-L2-Unit1-2015 | 645 | 0.9634 | 0.9461 |
| Makino-L3-Unit1-2015 | 660 | 0.9809 | 0.9937 |

Both models are scored on the same test rows, which matters — my first attempt at this comparison used different splits for the two, and the shared model looked artificially good because it had already seen most of those rows.

The separate model wins on two machines and loses on the third, 0.974 against 0.971 on average. With 158 to 179 test rows per machine, a difference of one point is two or three rows. It is a tie, and three models mean three things to train and maintain for no gain.

## What I would do next

1. Deploy it as a real-time check, not as a forecast.
2. Collect the data a forecast would need — anything that accumulates over time.
3. Retrain on recent data and compare against the time-split score, which gives a baseline for what healthy looks like on unseen later data.
4. Keep one model for all machines.

## Data

DataCamp guided project *Predicting Industrial Machine Downtime: Level 3*. 2500 rows, 16 columns, covering three Makino CNC machines from November 2021 to July 2022. Each row is one reading with twelve sensor measurements and a `Downtime` label. The dataset is not included in this repository.

## What is in here

```
.
├── README.md
├── notebooks/
│   └── machine_downtime.ipynb
├── requirements.txt
└── .gitignore
```

The notebook runs top to bottom and contains the full analysis with its output.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/machine_downtime.ipynb
```

The notebook expects the dataset at `data/machine_downtime.csv`.
