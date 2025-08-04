Link to download and transfer VA reports between Viya deployments.
To save report, open report, CTRL-ALT-B, save JSON format.
Create new report, CTRL-ALT-B, clear HTML & JSON, copy and paste JSON from saved file (may need to change caslibs in json file by searching "library"), Click 'Load'.
Once loaded, if visualizations do not populate, replace or refresh data sources that correspond to data items associated with each object.
If filters or other objects do not work, data types may need to be be tagged as "categories".

https://blogs.sas.com/content/sgf/2023/11/17/moving-sas-visual-analytics-reports-between-sas-viya-environments/