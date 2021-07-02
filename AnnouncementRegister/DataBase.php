<?php
require "DataBaseConfig.php";

class DataBase
{
    public $connect;
    public $data;
    private $sql;
    protected $servername;
    protected $anName;
    protected $location;
    protected $time_of_event;
    protected $no_Participants;
    protected $databasename;

    public function __construct()
    {
        $this->connect = null;
        $this->data = null;
        $this->sql = null;
        $dbc = new DataBaseConfig();
        $this->servername = $dbc->servername;
        $this->anName = $dbc->anName;
        $this->location = $dbc->location;
        $this->time_of_event = $dbc->time_of_event;
        $this->no_Participants = $dbc->no_Participants;
        $this->databasename = $dbc->databasename;
    }

    function dbConnect()
    {
        $this->connect = mysqli_connect($this->servername, $this->anName, $this->location, $this->time_of_event,$this->no_Participants,$this->databasename);
        return $this->connect;
    }

    function prepareData($data)
    {
        return mysqli_real_escape_string($this->connect, stripslashes(htmlspecialchars($data)));
    }

    function signUp($table, $fullname, $email, $username, $password)
    {

        $anName = $this->prepareData($anName);
        $location = $this->prepareData($location);
        $time_of_event = $this->prepareData($time_of_event);
        $no_Participants = $this->prepareData($no_participants);

        $this->sql =
            "INSERT INTO " . $table . " (fullname, username, password, email) VALUES ('" . $fullname . "','" . $username . "','" . $password . "','" . $email . "')";
        if (mysqli_query($this->connect, $this->sql)) {
            return true;
        } else return false;
    }

}

?>
