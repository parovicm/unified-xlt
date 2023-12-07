#!/bin/bash


################################################################################
#                                                                              #
# tidy_finish - remove working directory, etc                                  #
#                                                                              #
################################################################################

tidy_finish() {
   
   if [[ $DIAGNOSE -ne 1 && "$TMPDIR" = "/tmp/tmp."* ]]; then
      cd
      rm -rf "$TMPDIR"
      fi
   }

################################################################################
#                                                                              #
# abend(): exit with message to stderr                                         #
#                                                                              #
################################################################################

abend() {

   typeset msg

   msg="$1"

   echo "$msg" >&2

   tidy_finish

   exit 1
   }

################################################################################
#                                                                              #
# check_unique - check command line option is unique                           #
#                                                                              #
################################################################################

check_unique() {

   typeset varname
   typeset oldvalue
   typeset varvalue

   varname="$1"
   varvalue="$2"

   eval 'oldvalue="$'${varname}'"'
   if [[ "$oldvalue" != "" ]]; then
      abend "Duplicate or contradictory option specified in command line"
      fi

   eval ${varname}'="$varvalue"'
   }


################################################################################
#                                                                              #
# process_clparms - process command line parameters                            #
#                                                                              #
################################################################################

process_clparms() {

   local option
   local save=""

   CL_DRY_RUN=""
   CL_SLURM=""
   CL_SLURM_ARGS=""

   while [[ "$1" = -* ]]; do
      option="${1#\-}"
      case $option in
         d)
            check_unique CL_DRY_RUN "$option"
            ;;
         s)
             check_unique CL_SLURM "$option"
             save="CL_SLURM_ARGS"
             ;;
         *)
            abend "Invalid command line option specified: $1"
            ;;

         esac

      shift

      if [[ "$save" != "" ]]; then
          if [[ $# == 0 ]]; then
              abend "Missing value for option \"-${option}\""
          fi
          value="$1"
          eval $save="$value"
          shift
      fi

   done


   if [[ $# -lt 1 ]]; then
      abend "Missing batch file"
   fi
   CL_BATCH_FILE="$1"
   shift

   if [[ $# -ne 0 ]]; then
      abend "Unexpected or extraneous parameter(s) detected"
   fi

   if [[ "$CL_DRY_RUN" && ! "$CUDA_VISIBLE_DEVICES" ]]; then
       abend "Dry-run requires CUDA_VISIBLE_DEVICES to be set"
   fi
}

################################################################################
#                                                                              #
# get_tmp_dir(): get a private work directory and cd to it                     #
#                                                                              #
################################################################################

get_tmp_dir() {

   TMPDIR=$(mktemp -d)

   if [[ "$TMPDIR" != "/tmp/tmp."* ]]; then
      abend "Can't allocated a work directory"
      fi

   if [[ ! -d "$TMPDIR" ]]; then
      abend "Can't allocated a work directory"
      fi

}

init() {
    COMMAND_DIR="$PWD"
    BIN_DIR="${BASH_SOURCE[0]%/*}"

    get_tmp_dir
}

get_batch_id_and_dir() {
    BATCH_ID="${BATCH_NAME//\//_}_$(date +%Y-%m-%d_%H-%M-%S%3N)"
    BATCH_DIR="$JOBSCHED_BATCH/$BATCH_ID"

    if [[ -e "$BATCH_DIR" ]]; then
        abend "Batch directory $BATCH_DIR already exists"
    fi

    mkdir $BATCH_DIR
}   

interrupt_handler() {
    echo "Caught SIGINT"
    return
}

DIAGNOSE=0
init
process_clparms $*

if [[ ! -f $CL_BATCH_FILE ]]; then
    abend "No such file: $CL_BATCH_FILE"
fi

python $BIN_DIR/parse_batch.py < $CL_BATCH_FILE > $TMPDIR/parse.out 2> $TMPDIR/parse.err
if [[ $? != 0 ]]; then
    echo "Failed to parse batch description $BATCH_FILE:"
    cat $TMPDIR/parse.err
    tidy_finish
    exit 1
fi

{
    read BATCH_NAME SCRIPT

    if [[ "$CL_DRY_RUN" ]]; then
        echo "Will dry-run the following jobs:"
    else
        echo "Will submit the following jobs to Slurm:"
    fi
    echo "Experiment name: $BATCH_NAME"
    echo "Experiment script: $SCRIPT"

    while read JOB_NAME PARAMS; do
        echo "$JOB_NAME: $PARAMS"
    done
} < $TMPDIR/parse.out
echo "Do you wish to proceed? [Y/n]"

read CONFIRMATION
if [[ "$CONFIRMATION" != "y" && "$CONFIRMATION" != "Y" && "$CONFIRMATION" != "" ]]; then
    abend "Batch submission cancelled"
fi

if [[ ! -f "$SCRIPT" ]]; then
    abend "Run script $SCRIPT does not exist"
fi

{
    # Pass over params which have already been read
    read -u 3 _

    while read -u 3 JOB_NAME PARAMS; do
        OUTPUT_DIR="$JOB_NAME"
        if [[ "$CL_DRY_RUN" ]]; then
            echo "Dry run $JOB_NAME?. Enter Y to run, S to skip, X to exit."
            while true; do
                read CONFIRMATION
                if [[ "$CONFIRMATION" == [SsXxYy] ]]; then
                    break
                fi
                echo "Invalid option $CONFIRMATION"
            done

            if [[ "$CONFIRMATION" == [Xx] ]]; then
                break
            fi
            if [[ "$CONFIRMATION" == [Ss] ]]; then
                continue
            fi

            echo "Dry running $JOB_NAME"
            mkdir -p "$OUTPUT_DIR"
            if [[ $? != 0 ]]; then
                abend "Could not create job dir $OUTPUT_DIR"
            fi

            (
                export DIR="$OUTPUT_DIR"
                export $PARAMS
                trap "interrupt_handler" SIGINT
                bash $SCRIPT
                trap - SIGINT
            ) < /dev/null
            echo "Finished dry-running $JOB_NAME"
        else
            (
                export DIR="$OUTPUT_DIR"
                export $PARAMS
                sbatch $CL_SLURM_ARGS $SCRIPT
            )
        fi

    done
} 3<$TMPDIR/parse.out

tidy_finish

