function sumEven(array $numbers): int {
    $total = 0;
    foreach ($numbers as $n) {
        if ($n % 2 === 0) {
            $total += $n;
        }
    }
    return $total;
}
