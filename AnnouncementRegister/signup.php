<?php
require "DataBase.php";
$db = new DataBase();
if (isset($_POST['user']) && isset($_POST['location']) && isset($_POST['time_of_event']) && isset($_POST['no_Participants'])) {
    if ($db->dbConnect()) {
        if ($db->signUp("announcement", $_POST['user'], $_POST['location'], $_POST['time_of_event'], $_POST['no_Participants'])) {
            echo "Succesfully Added";
        } else echo "Sign up Failed";
    } else echo "Error: Database connection";
} else echo "All fields are required";
?>
