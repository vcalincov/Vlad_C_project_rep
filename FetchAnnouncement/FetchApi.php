<?php
    define('DB_HOST','localhost');
    define('DB_USER','root');
    define('DB_PASS','');
    define('DB_NAME','announcementregistry');

    $conn = new mysqli(DB_HOST,DB_USER,DB_PASS,DB_NAME);

    if(mysqli_connect_errno()){
        echo 'Unable to connect to database' . mysqli_connect_error();
        die();
    }

    $stmt = $conn->prepare("SELECT id, anName, location, time_of_event, no_Participants;");

    $stmt->execute();


$stmt->bind_result($id,$anName,$location,$time_of_event,$no_Participants);

    $announcement = array();

    while($stmt->fetch()){
        $temp = array();
        $temp['id'] = $id;
        $temp['anName'] = $anName;
        $temp['location'] = $location;
        $temp['time_of_event'] = $time_of_event;
        $temp['no_Particiapnts'] = $no_Participants;

        array_push($announcement, $temp);
    }

    echo json_encode($announcement);
