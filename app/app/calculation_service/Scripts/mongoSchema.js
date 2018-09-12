db.createCollection("expressions",{
    validator: {
        $and:
            [
                {
                    name: { $type: "string" },
                    expression: { $type: "string" },
                }
            ]
    },
validationAction: "warn"}
)