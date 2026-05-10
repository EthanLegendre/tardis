##
## EPITECH PROJECT, 2026
## config
## File description:
## config
##

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import sklearn as skl

## ----------------------- CONFIG VALUES --------------------------------

language = "English"
basic_dataset = "dataset.csv"
clean_dataset = "clean_dataset.csv"
separator = ";"

## ----------------------- CSV FILE COLUMNS -----------------------------

date = "Date"
service = "Service"
departure = "Departure station"
arrival = "Arrival station"
cancel_comment = "Cancellation comments"
departure_delay_comment = "Departure delay comments"
arrival_delay_comment = "Arrival delay comments"

number_train_sheduled = "Number of scheduled trains"
number_train_cancel = "Number of cancelled trains"
number_train_delayed_departure = "Number of trains delayed at departure"
number_train_delayed_arrival = "Number of trains delayed at arrival"
number_train_delay_sup_15min = "Number of trains delayed > 15min"
number_train_delay_sup_30min = "Number of trains delayed > 30min"
number_train_delay_sup_60min = "Number of trains delayed > 60min"

average_journey_time = "Average journey time"
average_delay_late_train_at_departure = "Average delay of late trains at departure"
average_delay_all_at_departure = "Average delay of all trains at departure"
average_delay_late_train_at_arrival = "Average delay of late trains at arrival"
average_delay_all_train_at_arrival = "Average delay of all trains at arrival"
average_delay_sup_15min_competing_flight = "Average delay of trains > 15min (if competing with flights)"

pct_delay_external_cause = "Pct delay due to external causes"
pct_delay_infrastructure = "Pct delay due to infrastructure"
pct_delay_traffic = "Pct delay due to traffic management"
pct_rolling_stock = "Pct delay due to rolling stock"
pct_equipement_station_management = "Pct delay due to station management and equipment reuse"
pct_passenger = "Pct delay due to passenger handling (crowding, disabled persons, connections)"

## ----------------------- FEATURE CSV FILE COLUMNS ---------------------

year = "Year"
month = "Month"
delay_category = "Delay category"
rate_cancel_train = "Rate Cancel train"
rate_delay_train = "Rate delay train"
is_holiday = "Is holiday"
go_to_Paris = "Go to Paris"
refundable = "Refundable"
