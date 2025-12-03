#!/usr/bin/env python
import os
import re
import math
import time
import shutil
import logging
import pathlib
import datetime
import subprocess
from tqdm import tqdm
from lxml import etree
from simple_slurm import Slurm
from shapely.geometry import Polygon

ALTO_NS = '{http://www.loc.gov/standards/alto/ns-v4#}'

logger = logging.getLogger(__name__)

class JobCancelled(Exception):
    "Custom exception for when slurm job was cancelled"

def dir_setup():
    pathlib.Path("../processing/imgs-completed").mkdir(parents=True, exist_ok=True)
    pathlib.Path("../processing/imgs-to-process").mkdir(parents=True, exist_ok=True)
    pathlib.Path("../processing/xml-completed").mkdir(parents=True, exist_ok=True)
    pathlib.Path("../processing/xml-to-process").mkdir(parents=True, exist_ok=True)

def file_setup():
    zipFiles =  [f for f in os.listdir("../1_UPLOAD") if ".zip" in f]
    if len(zipFiles) > 1:
        print("Warning: more than one .ZIP uploaded -- make sure this is intended!")
    elif len(zipFiles) == 0:
        raise FileNotFoundError("No .ZIP file(s) found to process in `1_UPLOAD/`!")
    
    for f in zipFiles:
        if 'pagexml' in f:
            raise TypeError("Error! This code expects ALTO XML files but `1_UPLOAD/` seems to contain PAGE XML exports!")
        shutil.unpack_archive(f"../1_UPLOAD/{f}", "../processing/imgs-to-process")
    
    os.remove("../processing/imgs-to-process/METS.xml")
    xmlFiles = [f for f in os.listdir("../processing/imgs-to-process") if ".xml" in f]
    for f in xmlFiles:
        shutil.move(f"../processing/imgs-to-process/{f}", f"../processing/xml-to-process/{f}")
    
def run_yaltai(
    repo_dir: pathlib.Path,
    time_requested: datetime.timedelta = datetime.timedelta(minutes=10),
    model: str = "model_citing-marx_2025-06.pt",
    email = False,
    ):
    
    if email:
        yaltai_slurm = Slurm(
            nodes=1,
            ntasks=1,
            cpus_per_task=1,
            mem_per_cpu='3G',
            job_name='yaltai',
            time=time_requested,
            mail_type='end',
            mail_user='EMAIL_HERE'
        )
    else:
        yaltai_slurm = Slurm(
            nodes=1,
            ntasks=1,
            cpus_per_task=1,
            mem_per_cpu='3G',
            job_name='yaltai',
            time=time_requested,
        )
    
    # add commands for setup steps
    yaltai_slurm.add_cmd("module purge")
    yaltai_slurm.add_cmd("module load anaconda3/2024.2")
    yaltai_slurm.add_cmd("conda activate yaltai")
    logger.info(f"sbatch file\n: {yaltai_slurm}")
    
    # sbatch returns the job id for the created job
    yaltai_cmd = (
        f'yaltai kraken --alto -I'
        + f' "{repo_dir}/processing/imgs-to-process/*"'
        + f' --suffix ".xml" segment --yolo "{repo_dir}/weights/{model}"'
    )

    logger.info(f"yaltai command: {yaltai_cmd}")
    return yaltai_slurm.sbatch(yaltai_cmd)
    
def move_completed():
    xml_completed = [f for f in os.listdir("../processing/imgs-to-process") if ".xml" in f]
    for f_xml in xml_completed:
        tree = etree.parse(f'../processing/imgs-to-process/{f_xml}')
        f_img = tree.find(f'.//{ALTO_NS}fileName').text.split('/')[-1]
        shutil.move(f"../processing/imgs-to-process/{f_img}", f"../processing/imgs-completed/{f_img}")
        shutil.move(f"../processing/imgs-to-process/{f_xml}", f"../processing/xml-completed/{f_xml}")
        
def calculate_remaining_time(job_stats):
    job_duration = re.findall('Run Time: (\d+:\d\d:\d\d)', job_stats)
    if job_duration:
        t = datetime.datetime.strptime(job_duration[0], '%H:%M:%S')
        job_duration = datetime.timedelta(hours=t.hour, minutes=t.minute, seconds=t.second).seconds
        
        count_completed = len(os.listdir("../processing/imgs-completed"))
        count_to_process = len(os.listdir("../processing/imgs-to-process"))
        count_total = count_completed + count_to_process
        
        total_duration = math.ceil(job_duration / count_completed * count_to_process * 1.2)
        duration_minutes = datetime.timedelta(minutes=math.ceil(total_duration / 60))
        
        # Don't put in a request for less than 10 minutes, or more than 10 hours
        if duration_minutes < datetime.timedelta(minutes=10):
            duration_minutes = datetime.timedelta(minutes=10)
        elif duration_minutes > datetime.timedelta(minutes=600):
            duration_minutes = datetime.timedelta(minutes=600)
        
        return duration_minutes
        
    else:
        return None
    
        
def parse_polygon(polygon_element):
    coords = polygon_element.attrib['POINTS'].split()
    points = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return Polygon(points)
    
def merge_xml():
    xmlFiles = [f for f in os.listdir('../processing/xml-completed') if '.xml' in f]
    for f in xmlFiles:
        tree = etree.parse(f'../processing/xml-completed/{f}')
        try:
            old_tree = etree.parse(f'../processing/xml-to-process/{f}')
        except:
            old_tree = None
    
        # fix fileName
        fileName = tree.find(f'.//{ALTO_NS}fileName')
        fileName.text = fileName.text.split('/')[-1]
    
        # grab regions from yaltai XML
        textBlocks = tree.findall(f'.//{ALTO_NS}TextBlock')
    
        # set up dummyblock if necessary
        printSpace = tree.find(f'.//{ALTO_NS}PrintSpace')
        dummyBlocks = tree.findall(f'.//{ALTO_NS}TextBlock[@ID="eSc_dummyblock_"]')
        if len(dummyBlocks) == 0:
            dummyBlock = etree.Element(f"{ALTO_NS}TextBlock")
            dummyBlock.attrib['ID'] = "eSc_dummyblock_"
            printSpace.append(dummyBlock)
        else:
            print(f"{len(dummyBlocks)} dummy block(s)")
            dummyBlock = dummyBlocks[0]
    
        if old_tree:
            # remove lines from yaltai XML
            for textBlock in textBlocks:
                textLines = textBlock.findall(f'.//{ALTO_NS}TextLine')
                for textLine in textLines:
                    textLine.getparent().remove(textLine)
            # find all lines in old kraken XML
            textLines = old_tree.findall(f'.//{ALTO_NS}TextLine')
    
            # match lines from old kraken XML to the regions from the yaltai XML which they share the greatest intersection with
            candidateDict = {}
            for textLine in textLines:
                poly = textLine.find(f'./{ALTO_NS}Shape/{ALTO_NS}Polygon')
                if poly is None:
                    continue
                polyLine = parse_polygon(poly)
                # print(polyLine)
                if polyLine is None:
                    continue
    
                best_match = None
                max_intersection_area = 0
    
                for textBlock in textBlocks:
                    poly2 = textBlock.find(f'./{ALTO_NS}Shape/{ALTO_NS}Polygon')
                    if poly2 is None:
                        continue
                    polyBlock = parse_polygon(poly2)
                    # print(polyBlock)
                    if polyBlock is None:
                        continue
    
                    intersection = polyLine.intersection(polyBlock)
                    if intersection.area > max_intersection_area:
                        max_intersection_area = intersection.area
                        best_match = textBlock
    
                if best_match is not None:
                    best_match.append(textLine)
                else:
                    dummyBlock.append(textLine)
    
        tree.write(f'../processing/xml-completed/{f}', encoding='utf-8', xml_declaration=True)
            
def make_download(now):
    shutil.make_archive(f'../2_DOWNLOAD/out_{now}', 'zip', '../processing/xml-completed/')
    
def reporting(report, msg):
    report = report + '\n' + msg
    print(msg)
    return report
    
##########################################################################################
# the below functions come from the PrincetonCDH htr2hpc library 
# https://github.com/Princeton-CDH/htr2hpc/

def slurm_job_queue_status(job_id: int) -> str:
    """Use `squeue` to get the full-word status (i.e., PENDING or RUNNING)
    for a queued slurm job."""
    result = subprocess.run(
        ["squeue", f"--jobs={job_id}", "--only-job-state", "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # return task status without any whitespace
    # squeue doesn't report on the task when it is completed and no longer in the queue,
    # so empty string means the job is complete
    return result.stdout.strip()


def slurm_job_status(job_id: int) -> set:
    """Use `sacct` to get the status of a slurm job that is no longer queued.
    Returns a set of unique full-word statuses, reporting across all tasks for the job.
    """
    result = subprocess.run(
        ["sacct", f"--jobs={job_id}", "--format=state%15", "--noheader"],
        capture_output=True,
        text=True,
    )
    # raise subprocess.CalledProcessError if return code indicates an error
    result.check_returncode()
    # sacct returns a table with status for each portion of the job;
    # return all unique status codes for now
    return set(result.stdout.split())
    
def slurm_job_stats(job_id: int) -> str:
    """Use `jobstats` to get Slurm Job Statistics, to track resource usage"""
    result = subprocess.run(
        ["jobstats", str(job_id)],
        capture_output=True,
        text=True,
    )
    # return task status without any whitespace
    return result.stdout.strip()

def monitor_slurm_job(job_id):
    # get initial job status (typically PENDING)
    job_status = slurm_job_queue_status(job_id)
    # typical states are PENDING, RUNNING, SUSPENDED, COMPLETING, and COMPLETED.
    # https://slurm.schedmd.com/job_state_codes.html
    # end states could be FAILED, CANCELLED, OUT_OF_MEMORY, TIMEOUT
    # * but note that squeue only reports on pending & running jobs

    # loop while the job is pending or running and then stop
    # use tqdm to display job status and wait time
    with tqdm(
        desc=f"Slurm job {job_id}",
        bar_format="{desc} | total time: {elapsed}{postfix} ",
    ) as statusbar:
        running = False
        runstart = time.time()
        while job_status:
            status = f"status: {job_status}"
            # display an unofficial runtime to aid in troubleshooting
            if running:
                runtime_elapsed = statusbar.format_interval(time.time() - runstart)
                status = f"{status}  ~ run time: {runtime_elapsed}"
            statusbar.set_postfix_str(status)
            time.sleep(1)
            job_status = slurm_job_queue_status(job_id)
            # capture start time first time we get a status of running
            if not running and job_status == "RUNNING":
                running = True
                runstart = time.time()

    # check the completed status
    job_status = slurm_job_status(job_id)
    print(
        f"Job {job_id} is no longer queued; ending status: {','.join(job_status)}"
    )

    # when cancelled via delete button on myadroit web ui,
    # statuses are COMPLETED,CANCELLED
    # if time limit ran out, status will include TIMEOUT as well as CANCELLED
    if "CANCELLED" in job_status and "TIMEOUT" not in job_status:
        raise JobCancelled

##########################################################################################

def main():
    repo_dir = pathlib.Path.cwd().parent
    now = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    report = ""
    wip_delete = True
    
    try:
        # setup
        dir_setup()
        file_setup()
        
        report = reporting(report, "Running first calibration job...")
        
        files_to_process = len(os.listdir('../processing/imgs-to-process'))
        if files_to_process == 0:
            raise ValueError("Error: no images to process! Did the eScr export include images?")
        
        # run the first job
        job_id = run_yaltai(
            repo_dir,
            datetime.timedelta(minutes=10),
            "model_citing-marx_2025-06.pt",
        )
        monitor_slurm_job(job_id)
        
        # determine how much time is needed for the full job and submit it
        job_stats = slurm_job_stats(job_id)
        move_completed()
        
        files_completed = len(os.listdir('../processing/imgs-completed'))
        files_to_process = len(os.listdir('../processing/imgs-to-process'))
        report = reporting(report, "Calibration job completed.")
        report = reporting(report, f"{files_completed} images processed. {files_to_process} images remain.")

        tries = 0
        while files_to_process > 0 and tries < 2:
            tries = tries + 1
            time_requested = calculate_remaining_time(job_stats)
            report = reporting(report, f"Submitting a job with a duration of {time_requested} minutes...")
            
            job_id = run_yaltai(
                repo_dir,
                time_requested,
                "model_citing-marx_2025-06.pt",
                True,
            )
            monitor_slurm_job(job_id)
            
            job_stats = slurm_job_stats(job_id)
            move_completed()
            
            files_completed = len(os.listdir('../processing/imgs-completed'))
            files_to_process = len(os.listdir('../processing/imgs-to-process'))
            report = reporting(report, "Job completed.")
            report = reporting(report, f"{files_completed} images processed. {files_to_process} images remain.")
        
        files_to_process = len(os.listdir('../processing/imgs-to-process'))
        if files_to_process > 0:
            report = reporting(report, f"Warning: {files_to_process} images are still unprocessed!")
            report = reporting(report, "This code will not automatically delete the temporary `processing/` directory but will instead leave it untouched for error checking.")
            wip_delete = False
        
        # merge output files and create the ZIP file for download
        report = reporting(report, "Merging XML files...")
        merge_xml()
        report = reporting(report, "Creating the ZIP file to download...")
        make_download(now)
        report = reporting(report, f"Results in 2_DOWNLOAD/out_{now}.zip.")
        
        # cleanup the uploaded ZIP file(s)
        report = reporting(report, "Deleting uploaded zip files...")
        zipFiles = [f for f in os.listdir("../1_UPLOAD") if '.zip' in f]
        for f in zipFiles:
            os.remove(f"../1_UPLOAD/{f}")
            
        report = reporting(report, "Done!")
        
    except Exception as err:
        report = reporting(report, f"Something went wrong: {err}")
    
    # always clean up the processing directory unless flagged otherwise
    if wip_delete:
        shutil.rmtree("../processing")
        
    # always save a report of progress
    with open(f'../2_DOWNLOAD/report_{now}.txt', 'w') as outFile:
        outFile.write(report)

if __name__ == "__main__":
    main()